"""
ResVLA with Intent Anchoring and Residual Bridge.
"""
from typing import List, Optional, Tuple
from pathlib import Path
import sys
import torch
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resVLA.training.trainer_utils import initialize_overwatch
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100

from resVLA.model.framework.base_framework import baseframework
from resVLA.model.modules.vlm import get_vlm_model
from resVLA.model.modules.vlm.QWen3 import _QWen3_VL_Interface
from resVLA.model.modules.action_model.ResVLA_ActionHead import get_resVLA_action_model, ResVLAActionHead
from resVLA.training.trainer_utils.trainer_tools import resize_images
from resVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("ResVLA")
class QwenResVLA(baseframework):
    """
    Components:
      - Qwen3-VL: vision-language backbone
      - ResVLAActionHead: Intent Anchoring + Residual Bridge action head
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the QwenResVLA model.

        Args:
            config: Model configuration containing:
                - framework.qwenvl: VLM configuration
                - framework.action_model: ResVLAActionHead configuration
        """
        super().__init__()
        self.config = config

        # 1. Initialize the VLM interface
        self.qwen_vl_interface: _QWen3_VL_Interface = get_vlm_model(config=self.config)

        # 2. Fetch VLM-derived configuration values dynamically
        num_vl_layers = self.qwen_vl_interface.model.config.text_config.num_hidden_layers
        llm_hidden_size = self.qwen_vl_interface.model.config.hidden_size
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        # 3. Initialize the action model
        self.action_model: ResVLAActionHead = get_resVLA_action_model(config=self.config)

        # 4. Cache frequently used configuration values
        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        logger.info("QwenResVLA initialized successfully")
        logger.info(f"   - VLM layers: {num_vl_layers}")
        logger.info(f"   - Total DiT layers: {self.action_model.total_dit_layers}")
        logger.info(
            f"   - Architecture: {self.action_model.num_anchor_layers} anchor layers + "
            f"{self.action_model.num_bridge_layers} bridge layers"
        )


    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Training forward pass.

        Flow:
          1. Extract data (images, instructions, actions, state)
          2. Run the VLM forward pass -> vl_embs_list
          3. Run the action model forward pass

        Args:
            examples: List of training samples. Each sample contains:
                - image: list of images (multi-view)
                - lang: instruction string
                - action: action sequence of shape (T, action_dim)
                - state: optional robot state

        Returns:
            dict: {"action_loss": loss}
        """
        # ========== 1. Data preparation ==========
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        # ========== 2. VLM forward pass ==========
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            all_hidden = qwenvl_outputs.hidden_states
            vl_embs_list = list(all_hidden[1:])
            base_hidden = vl_embs_list[-1]

        # ========== 3. Action model forward pass ==========
        with torch.autocast("cuda", dtype=torch.float32):
            # Convert actions to tensors and crop to the target horizon
            actions = torch.tensor(
                np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype
            )
            actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 2) if self.config and self.config.trainer else 2
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            # Repeat features from each VLM layer
            vl_embs_list_repeated = [h.repeat(repeated_diffusion_steps, 1, 1) for h in vl_embs_list]

            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=base_hidden.device, dtype=base_hidden.dtype
                )
                state_repeated = state.unsqueeze(1).repeat(repeated_diffusion_steps, 1, 1)

            action_loss, intent_anchor_loss = self.action_model(vl_embs_list_repeated, actions_target_repeated, state_repeated)

        return {"action_loss": action_loss, "intent_anchor_loss": intent_anchor_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Prediction flow:
          1. Extract data and resize images
          2. Run the VLM forward pass -> vl_embs_list
          3. Run action model inference

        Args:
            examples: List of inference samples

        Returns:
            dict: {"normalized_actions": np.ndarray}
        """
        # ========== 1. Preprocessing ==========
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # ========== 2. VLM forward pass ==========
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            all_hidden = qwenvl_outputs.hidden_states
            # ResVLA uses all transformer layers for the intent-anchor and residual-bridge stages.
            # all_hidden[0] is the embedding output, and all_hidden[1:] are transformer layer outputs.
            vl_embs_list = list(all_hidden[1:])  
            base_hidden = vl_embs_list[-1]

        # ========== 3. Action model inference ==========
        state = torch.from_numpy(np.array(state)).unsqueeze(1).to(base_hidden.device, dtype=base_hidden.dtype) if state is not None else None

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(vl_embs_list, state)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}


if __name__ == "__main__":
    # Test code
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./examples/LIBERO/train_files/resvla_cotrain_libero.yaml", help="Path to YAML config.")
    parser.add_argument("--base_vlm", type=str, default=None, help="Optional VLM checkpoint override.")
    args, _ = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    cfg.framework.name = "ResVLA"
    if args.base_vlm is not None:
        cfg.framework.qwenvl.base_vlm = args.base_vlm

    # Create model
    model = QwenResVLA(cfg)
    print("✅ QwenResVLA model created successfully")

    # Test forward pass
    action_dim = int(cfg.framework.action_model.action_dim)
    state_dim = int(cfg.framework.action_model.get("state_dim", action_dim))
    action_horizon = int(
        cfg.framework.action_model.get(
            "action_horizon",
            cfg.framework.action_model.future_action_window_size + 1,
        )
    )
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(action_horizon, action_dim)).astype(np.float32),
        "image": [image, image],
        "lang": "Test instruction",
        "state": np.random.uniform(-1, 1, size=(state_dim,)).astype(np.float32),
    }

    batch = [sample, sample]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    output = model(batch)
    print(f"Loss: {output['action_loss'].item()}")
    # test predict action
    predict_output = model.predict_action([sample])
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    # from resVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    # vla_dataset_cfg = cfg.datasets.vla_data
    # dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    # from torch.utils.data import DataLoader

    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=2,
    #     num_workers=1,  # For Debug
    #     collate_fn=collate_fn,
    # )
    # #
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     batch
    #     break

    # # try get model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model(batch)

    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]])

    # # fake state
    # for ba in batch:
    #     ba["state"] = ba["action"][0][None]

    # model(batch)
    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])

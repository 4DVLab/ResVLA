import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from torch_dct import LinearDCT
from transformers.feature_extraction_utils import BatchFeature
from resVLA.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)
from resVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT


class IntentAnchorHead(nn.Module):
    """
    Intent Anchor MLP that predicts low-frequency action anchors from intent queries.
    """
    def __init__(self, query_length, hidden_dim, action_horizon, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(query_length * hidden_dim),
            nn.Linear(query_length * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_horizon * action_dim),
            nn.LayerNorm(action_horizon * action_dim),
        )
    
    def forward(self, info_query_output):
        """
        Args:
            info_query_output: (bs, query_length, hidden_dim)
        Returns:
            intent_anchor_mean: (bs, action_horizon * action_dim)
        """
        return self.model(info_query_output)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, output_dim=2048):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size=1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        a_emb = self.layer1(actions)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))
        x = self.layer3(x)
        return x


class FrequencyCutoffSelector(nn.Module):
    def __init__(self, num_freq_components, hidden_dim):
        super().__init__()
        self.num_freq_components = num_freq_components
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_freq_components),
        )
    def forward(self, x, freq_action, mode="train"):
        logits = self.model(x)
        tau = 1.0
        if mode == "train":
            one_hot = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        else:
            one_hot = F.one_hot(logits.argmax(dim=-1), num_classes=logits.shape[-1]).float()
        mask = torch.flip(torch.cumsum(torch.flip(one_hot, dims=[-1]), dim=-1), dims=[-1])
        mask = mask.transpose(1, 2)
        mask = mask.expand_as(freq_action)
        masked_freq_action = freq_action * mask
        return masked_freq_action, mask



class ResVLAActionHead(nn.Module):
    
    def __init__(self, global_config, **kwargs):
        super().__init__()
        action_config = global_config.framework.action_model
        diffusion_model_cfg = action_config.diffusion_model_cfg
        
        DiTConfig = {
            "num_layers": global_config.framework.qwenvl.num_vl_layers,
            "input_embedding_dim": global_config.framework.qwenvl.vl_hidden_dim,
            "attention_head_dim": global_config.framework.action_model.diffusion_model_cfg.attention_head_dim,
            "output_dim": global_config.framework.qwenvl.vl_hidden_dim,
        }
        DiTConfig["num_attention_heads"] = DiTConfig["input_embedding_dim"] // DiTConfig["attention_head_dim"]
        diffusion_model_cfg.update(DiTConfig)
        diffusion_model_cfg.cross_attention_dim = DiTConfig["input_embedding_dim"]
        
        self.input_embedding_dim = global_config.framework.qwenvl.vl_hidden_dim
        self.model = DiT(**diffusion_model_cfg)
        self.dit_out_hidden_size = self.input_embedding_dim
        self.action_dim = action_config.action_dim
        self.action_horizon = action_config.future_action_window_size + 1
        self.num_inference_timesteps = action_config.num_inference_timesteps
        self.anchor_train_noise_std = float(action_config.get("anchor_train_noise_std", 1))
        self.anchor_infer_noise_std = float(
            action_config.get("anchor_infer_noise_std", self.anchor_train_noise_std)
        )
        
        self.dct_layer = LinearDCT(self.action_horizon, type='dct', norm='ortho')
        self.idct_layer = LinearDCT(self.action_horizon, type='idct', norm='ortho')
        
        for param in self.dct_layer.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        for param in self.idct_layer.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        
        self.intent_queries = nn.Embedding(action_config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.intent_queries.weight, mean=0.0, std=0.02)
        
        self.intent_anchor_head = IntentAnchorHead(
            action_config.num_target_vision_tokens,
            self.input_embedding_dim,
            self.action_horizon,
            self.action_dim
        )
        self.frequency_cutoff_token = nn.Embedding(1, self.input_embedding_dim)
        nn.init.normal_(self.frequency_cutoff_token.weight, mean=0.0, std=0.02)
        self.frequency_cutoff_selector = FrequencyCutoffSelector(self.action_horizon, self.input_embedding_dim)
        
        self.anchor_layer_ratio = action_config.get("anchor_layer_ratio", 0.85)
        self.num_anchor_layers = int(DiTConfig["num_layers"] * self.anchor_layer_ratio)
        self.num_bridge_layers = DiTConfig["num_layers"] - self.num_anchor_layers
        self.total_dit_layers = self.num_anchor_layers + self.num_bridge_layers
        self.state_encoder = MLP(
            input_dim=action_config.state_dim,
            output_dim=self.input_embedding_dim,
        ) if action_config.state_dim else None
        
        self.action_encoder = ActionEncoder(
            action_dim=action_config.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        
        self.action_decoder = MLP(
            input_dim=self.input_embedding_dim,
            hidden_dim=1024,
            output_dim=self.action_dim,
        )
        
        if action_config.add_pos_embed:
            self.position_embedding = nn.Embedding(action_config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        
        self.beta_dist = Beta(action_config.noise_beta_alpha, action_config.noise_beta_beta)
        self.num_timestep_buckets = action_config.num_timestep_buckets
        self.config = action_config
        

    def sample_time(self, batch_size, device, dtype):
        """Sample timesteps from the Beta distribution."""
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return self.config.noise_s * (1-sample)

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)
    
    def _process_output(self, hidden_states, temb, actions_length):
        """
        Post-process transformer outputs with adaptive layer normalization (AdaLN).
        
        Args:
            hidden_states: Tensor of shape (B, seq_length, embedding_dim)
            temb: Tensor of shape (B, embedding_dim) containing timestep embeddings
            actions_length: Length of the actions sequence (T)
        
        Returns:
            pred_velocity: Tensor of shape (B, T, action_dim)
        """
        conditioning = temb
        shift, scale = self.model.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.model.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        action_features = self.model.proj_out_2(hidden_states)
        pred = self.action_decoder(action_features)
        pred_velocity = pred[:, -actions_length:]
        return pred_velocity
    
    def forward(self, vl_embs_list: list, actions: torch.Tensor, state: torch.Tensor = None):
        """
        Args:
            vl_embs_list: Hidden states from all VLM layers
            actions: Time-domain actions of shape (B, action_horizon, action_dim)
            state: Optional robot state tensor of shape (B, 1, state_dim)
        Returns:
            loss: Flow matching loss
        """
        device = actions.device
        B = actions.shape[0]

        freq_actions_full = self.dct_layer(actions.transpose(1, 2)).transpose(1, 2)
        state_features = self.state_encoder(state) if state is not None else None
        intent_queries = self.intent_queries.weight.unsqueeze(0).expand(B, -1, -1)
        frequency_cutoff_query = self.frequency_cutoff_token.weight.unsqueeze(0).expand(B, -1, -1)
        if state_features is not None:
            intent_anchor_input = torch.cat((state_features, intent_queries, frequency_cutoff_query), dim=1)
        else:
            intent_anchor_input = torch.cat((intent_queries, frequency_cutoff_query), dim=1)

        intent_anchor_vl_embs = vl_embs_list[:self.num_anchor_layers]
        intent_anchor_output = intent_anchor_input
        dummy_timestep = torch.ones(B, dtype=torch.long, device=device) * self.num_timestep_buckets
        intent_anchor_temb = self.model.timestep_encoder(dummy_timestep)
        
        for layer_idx in range(self.num_anchor_layers):
            intent_anchor_output = self.model.transformer_blocks[layer_idx](
                hidden_states=intent_anchor_output,
                encoder_hidden_states=intent_anchor_vl_embs[layer_idx],
                temb=intent_anchor_temb
            )
        
        state_len = state_features.shape[1] if state_features is not None else 0
        intent_query_output = intent_anchor_output[:, state_len:state_len + intent_queries.shape[1], :]
        state_output = intent_anchor_output[:, :state_len, :] if state_features is not None else None

        intent_anchor_freq_flat = self.intent_anchor_head(intent_query_output)
        intent_anchor_freq_flat = intent_anchor_freq_flat.reshape(B, self.action_horizon, self.action_dim)

        output_freq_token = intent_anchor_output[:, -1, :].unsqueeze(1)
        intent_anchor_freq_full, freq_mask = self.frequency_cutoff_selector(output_freq_token, intent_anchor_freq_flat, "train")

        intent_anchor_loss = (((freq_actions_full - intent_anchor_freq_flat) ** 2) * freq_mask).sum() / (freq_mask.sum() + 1e-6)
        intent_anchor_time = self.idct_layer(intent_anchor_freq_full.transpose(1, 2)).transpose(1, 2)
        noise = self.anchor_train_noise_std * torch.randn(
            actions.shape, device=device, dtype=actions.dtype
        )
        x_0 = intent_anchor_time + noise
        x_1 = actions
        
        t = self.sample_time(B, device, actions.dtype)
        t_expanded = t[:, None, None]
        noisy_trajectory = (1 - t_expanded) * x_0 + t_expanded * x_1
        velocity = x_1 - x_0
        t_discretized = (t * self.num_timestep_buckets).long()
        noisy_action_features = self.action_encoder(noisy_trajectory, t_discretized)
        if self.config.add_pos_embed:
            pos_ids = torch.arange(noisy_action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            noisy_action_features = noisy_action_features + pos_embs
        
        diffusion_input = noisy_action_features
        temb = self.model.timestep_encoder(t_discretized)
        if state_output is not None:
            diffusion_vl_embs = [
                torch.cat((state_output, vl_emb), dim=1)
                for vl_emb in vl_embs_list[self.num_anchor_layers:]
            ]
        else:
            diffusion_vl_embs = vl_embs_list[self.num_anchor_layers:]
        diffusion_bridge_output = diffusion_input
        
        for layer_idx in range(self.num_anchor_layers, len(self.model.transformer_blocks)):
            vl_idx = layer_idx - self.num_anchor_layers
            diffusion_bridge_output = self.model.transformer_blocks[layer_idx](
                hidden_states=diffusion_bridge_output,
                encoder_hidden_states=diffusion_vl_embs[vl_idx],
                temb=temb,
            )
        pred_velocity = self._process_output(diffusion_bridge_output, temb, self.action_horizon)
        fm_loss = ((pred_velocity - velocity) ** 2).mean()

        loss = fm_loss + (fm_loss.detach() / (intent_anchor_loss.detach() + 1e-8)) * 0.25 * intent_anchor_loss
        return loss, intent_anchor_loss
    
    @torch.no_grad()
    def predict_action(self, vl_embs_list: list, state: torch.Tensor = None) -> torch.Tensor:
        """
        Inference-time action prediction with two-stage sampling.
        
        Args:
            vl_embs_list: Hidden states from all VLM layers
            state: Optional robot state
        Returns:
            actions: Predicted time-domain actions of shape (B, action_horizon, action_dim)
        """
        batch_size = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device
        dtype = vl_embs_list[0].dtype
        
        state_features = self.state_encoder(state) if state is not None else None
        intent_queries = self.intent_queries.weight.unsqueeze(0).expand(batch_size, -1, -1)
        frequency_cutoff_query = self.frequency_cutoff_token.weight.unsqueeze(0).expand(batch_size, -1, -1)
        if state_features is not None:
            intent_anchor_input = torch.cat((state_features, intent_queries, frequency_cutoff_query), dim=1)
        else:
            intent_anchor_input = torch.cat((intent_queries, frequency_cutoff_query), dim=1)
        intent_anchor_vl_embs = vl_embs_list[:self.num_anchor_layers]
        intent_anchor_output = intent_anchor_input
        dummy_timestep = torch.ones(batch_size, dtype=torch.long, device=device) * self.num_timestep_buckets
        intent_anchor_temb = self.model.timestep_encoder(dummy_timestep)
        
        for layer_idx in range(self.num_anchor_layers):
            intent_anchor_output = self.model.transformer_blocks[layer_idx](
                hidden_states=intent_anchor_output,
                encoder_hidden_states=intent_anchor_vl_embs[layer_idx],
                temb=intent_anchor_temb,
            )
        
        state_len = state_features.shape[1] if state_features is not None else 0
        intent_query_output = intent_anchor_output[:, state_len:state_len + intent_queries.shape[1], :]
        state_output = intent_anchor_output[:, :state_len, :] if state_features is not None else None
        output_freq_token = intent_anchor_output[:, -1, :].unsqueeze(1)
        intent_anchor_freq_flat = self.intent_anchor_head(intent_query_output)
        intent_anchor_freq_flat = intent_anchor_freq_flat.reshape(batch_size, self.action_horizon, self.action_dim)

        intent_anchor_freq_full, _ = self.frequency_cutoff_selector(output_freq_token, intent_anchor_freq_flat, "eval")
        intent_anchor_time = self.idct_layer(intent_anchor_freq_full.transpose(1, 2)).transpose(1, 2)
        noise = self.anchor_infer_noise_std * torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=dtype,
            device=device,
        )
        time_actions = intent_anchor_time + noise
        
        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps
        
        if state_output is not None:
            diffusion_vl_embs = [
                torch.cat((state_output, vl_emb), dim=1)
                for vl_emb in vl_embs_list[self.num_anchor_layers:]
            ]
        else:
            diffusion_vl_embs = vl_embs_list[self.num_anchor_layers:]
        
        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized_int = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized_int, device=device, dtype=torch.long
            )
            
            noisy_action_features = self.action_encoder(time_actions, timesteps_tensor)
            
            if self.config.add_pos_embed:
                pos_ids = torch.arange(noisy_action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                noisy_action_features = noisy_action_features + pos_embs
            
            diffusion_input = noisy_action_features
            temb = self.model.timestep_encoder(timesteps_tensor)
            diffusion_bridge_output = diffusion_input
            for layer_idx in range(self.num_anchor_layers, len(self.model.transformer_blocks)):
                vl_idx = layer_idx - self.num_anchor_layers
                diffusion_bridge_output = self.model.transformer_blocks[layer_idx](
                    hidden_states=diffusion_bridge_output,
                    encoder_hidden_states=diffusion_vl_embs[vl_idx],
                    temb=temb,
                )
            pred_velocity = self._process_output(diffusion_bridge_output, temb, self.action_horizon)
            time_actions = time_actions + dt * pred_velocity
        actions = time_actions
        
        return actions
    
    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def get_resVLA_action_model(config=None):
    """
    Factory function that constructs a ResVLAActionHead instance.
    
    Args:
        config: Global configuration
    Returns:
        ResVLAActionHead
    """
    return ResVLAActionHead(global_config=config)

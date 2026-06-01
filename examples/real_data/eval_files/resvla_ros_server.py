import argparse
import base64
import json
import sys
import time
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import rospy
import torch
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Header, String

from resVLA.model.framework.base_framework import baseframework
from resVLA.model.tools import read_mode_config



def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def validate_scale(name: str, values: list[float], expected_dim: int = 14) -> np.ndarray:
    scale = np.asarray(values, dtype=np.float32)
    if scale.shape[0] != expected_dim:
        raise ValueError(f"{name} must contain {expected_dim} values, got {scale.shape[0]}.")
    return scale


def load_policy(checkpoint_path: str, device: str):
    model = baseframework.from_pretrained(checkpoint_path)
    model = model.to(device)
    model.eval()
    return model


def decode_image_from_b64(encoded: str) -> np.ndarray:
    raw = base64.b64decode(encoded)
    img_buf = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image from observation payload.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def parse_observation_payload(payload: str) -> dict:
    # Preferred format: plain JSON string.
    try:
        return json.loads(payload)
    except Exception:
        pass

    # Common case when a bytes repr is accidentally stringified, e.g. "b'...'"
    normalized = payload.strip()
    if normalized.startswith("b'") and normalized.endswith("'"):
        normalized = normalized[2:-1]
    if normalized.startswith('b"') and normalized.endswith('"'):
        normalized = normalized[2:-1]

    # Compatibility: whole payload is base64(JSON).
    try:
        decoded = base64.b64decode(normalized).decode("utf-8")
        return json.loads(decoded)
    except Exception:
        pass

    # URL-safe base64 fallback.
    try:
        decoded = base64.urlsafe_b64decode(normalized).decode("utf-8")
        return json.loads(decoded)
    except Exception as exc:
        preview = normalized[:160]
        raise ValueError(f"Invalid observation payload. Preview: {preview}") from exc


def unnormalize_action_chunk(normalized_actions: np.ndarray, action_norm_stats) -> np.ndarray:
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
    action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
    normalized_actions = np.clip(normalized_actions, -1, 1)
    # normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )

    return actions


@torch.inference_mode()
def infer_action_chunk(model, example: dict) -> np.ndarray:
    output = model.predict_action(examples=[example])
    normalized_actions = output["normalized_actions"][0]
    return np.asarray(normalized_actions, dtype=np.float32)


class ResVLAPolicyServer:
    def __init__(self, args):
        self.args = args
        self.obs_queue = deque(maxlen=5)
        self.cached_action_chunk = None
        self.step_in_chunk = 0
        self.cur_state = np.zeros(14, dtype=np.float32)
        self.chunk_reference_state = None
        self.action_scale = validate_scale("action_scale", args.action_scale)
        self.state_scale = validate_scale("state_scale", args.state_scale)

        self.pred_action_publisher = None
        self.pred_chunk_publisher = None
        self.model = load_policy(args.checkpoint_path, args.device)
        self.action_norm_stats = self.get_stats(args.checkpoint_path, args.unnorm_key, "action")
        self.state_norm_stats = self.get_stats(args.checkpoint_path, args.unnorm_key, "state") if args.use_state else None
        self.action_chunk_size = self.get_action_chunk_size(args.checkpoint_path)
        self.init_ros()
        self._maybe_disable_framework_resize()

    def get_stats(self, policy_ckpt_path, unnorm_key, norm_type):
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)  # read config and norm_stats

        unnorm_key = self._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key][norm_type]

    def get_action_chunk_size(self, policy_ckpt_path):
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, _ = read_mode_config(policy_ckpt_path)
        return model_config["framework"]["action_model"]["future_action_window_size"] + 1
    
    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """
        Duplicate helper (retained for backward compatibility).
        See primary _check_unnorm_key above.
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key


    def _maybe_disable_framework_resize(self):
        if self.args.enable_model_resize:
            rospy.loginfo("Keeping framework-internal image resize.")
            return
        try:
            datasets_cfg = getattr(self.model.config, "datasets", None)
            vla_data_cfg = getattr(datasets_cfg, "vla_data", None) if datasets_cfg is not None else None
            if vla_data_cfg is not None and hasattr(vla_data_cfg, "image_size"):
                vla_data_cfg.image_size = None
            rospy.loginfo("Disabled framework-internal image resize. Image size is controlled by ros_action_to_aloha.")
        except Exception as exc:
            rospy.logwarn(f"Failed to disable framework resize: {exc}")

    def init_ros(self):
        rospy.init_node("resvla_policy_server", anonymous=True)
        rospy.Subscriber(self.args.obs_topic, String, self.obs_callback, queue_size=10, tcp_nodelay=True)
        self.pred_action_publisher = rospy.Publisher(self.args.pred_action_topic, JointState, queue_size=10)
        self.pred_chunk_publisher = rospy.Publisher(self.args.pred_action_chunk_topic, Float64MultiArray, queue_size=4)
        if self.args.use_delta:
            rospy.loginfo(
                "Chunk-wise delta control enabled: each predicted chunk is decoded against the chunk-start state."
            )

    def obs_callback(self, msg: String):
        self.obs_queue.append(msg.data)

    def get_latest_observation(self):
        if len(self.obs_queue) == 0:
            return None
        payload = self.obs_queue[-1]
        self.obs_queue.clear()
        return payload

    def build_example(self, payload: str) -> dict:
        obs = parse_observation_payload(payload)
        image_dict = obs.get("images", {})
        prompt = obs.get("prompt", None)
        raw_state = obs.get("state", None)
        if not isinstance(prompt, str) or not prompt.strip():
            raise KeyError("Observation payload must contain non-empty `prompt` from ros_action_to_aloha.")

        decoded_images = {}
        for camera_name, encoded_img in image_dict.items():
            decoded_images[camera_name] = decode_image_from_b64(encoded_img)

        selected_images = []
        for camera_name in self.args.policy_cameras:
            if camera_name not in decoded_images:
                raise KeyError(f"Missing camera `{camera_name}` in observation payload.")
            selected_images.append(decoded_images[camera_name])

        if self.args.use_state:
            if not isinstance(raw_state, list):
                raise KeyError("Observation payload must contain list object `state` from ros_action_to_aloha")
            raw_state = np.asarray(raw_state, dtype=np.float32)
            if raw_state.shape[0] != self.state_scale.shape[0]:
                raise ValueError(f"Expected state dim {self.state_scale.shape[0]}, got {raw_state.shape[0]}.")
            raw_state = raw_state / self.state_scale
            self.cur_state = raw_state
            # norm
            state_min, state_max = np.array(self.state_norm_stats['min']), np.array(self.state_norm_stats['max'])
            mask = self.state_norm_stats.get("mask", np.ones_like(self.state_norm_stats["min"], dtype=bool))
            state = np.where(
                mask,
                2 * (raw_state - state_min) / (state_max - state_min) - 1,
                raw_state
            )
            return {
                "image": selected_images,
                "lang": prompt.strip(),
                "state": state
            }
        else:
            return {
                "image": selected_images,
                "lang": prompt.strip(),
            }

    def publish_action(self, action: np.ndarray):
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.name = [f"joint{i}" for i in range(action.shape[0])]
        msg.position = action.tolist()
        self.pred_action_publisher.publish(msg)

    def publish_action_chunk(self, action_chunk: np.ndarray):
        msg = Float64MultiArray()
        msg.data = action_chunk.reshape(-1).tolist()
        self.pred_chunk_publisher.publish(msg)

    def run(self):
        loop_rate = rospy.Rate(self.args.control_hz)
        publish_loop = rospy.Rate(self.args.publish_rate)

        rospy.loginfo("ResVLA ROS inference server is ready. Waiting for observation topic...")
        while not rospy.is_shutdown():
            payload = self.get_latest_observation()
            if payload is None:
                rospy.loginfo_throttle(5.0, f"Waiting for observation topic: {self.args.obs_topic}")
                loop_rate.sleep()
                continue

            try:
                start_time = time.time()
                example = self.build_example(payload)
                if self.cached_action_chunk is None or self.step_in_chunk % self.action_chunk_size == 0:
                    # Chunk-wise delta control: lock the reference state once per chunk.
                    if self.args.use_delta:
                        self.chunk_reference_state = self.cur_state.copy()
                    normalized_actions = infer_action_chunk(self.model, example)
                    self.cached_action_chunk = unnormalize_action_chunk(normalized_actions, self.action_norm_stats)
                    self.action_chunk_size = len(self.cached_action_chunk)
                    self.publish_action_chunk(self.cached_action_chunk)

                action = self.cached_action_chunk[self.step_in_chunk % self.action_chunk_size]
                if self.args.use_delta:
                    if self.chunk_reference_state is None:
                        raise RuntimeError("Chunk reference state is not initialized for delta decoding.")
                    action = np.where(
                        self.action_norm_stats.get("mask", np.ones_like(self.action_norm_stats["min"], dtype=bool)),
                        action + self.chunk_reference_state,
                        action,
                    )
                action = action * self.action_scale
                self.publish_action(action)
                self.step_in_chunk += 1

                rospy.loginfo_throttle(
                    2.0,
                    f"Action step {self.step_in_chunk} / chunk {self.action_chunk_size}, latency={time.time() - start_time:.3f}s",
                )
                publish_loop.sleep()
            except Exception as exc:
                rospy.logerr_throttle(5.0, f"Inference loop error, keep running: {exc}")
                loop_rate.sleep()
                continue

            if self.args.chunk_sleep > 0:
                time.sleep(self.args.chunk_sleep)
            loop_rate.sleep()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to ResVLA checkpoint .pt file.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device, e.g. cuda or cpu.")
    parser.add_argument(
        "--policy_cameras",
        type=lambda s: [item.strip() for item in s.split(",") if item.strip()],
        default="cam_front,cam_high,cam_left,cam_right",
        help="Comma-separated camera order fed into ResVLA.",
    )
    parser.add_argument("--control_hz", type=int, default=10, help="Inference loop rate.")
    parser.add_argument("--publish_rate", type=int, default=30, help="Rate for publishing each action in the predicted chunk.")
    parser.add_argument("--chunk_sleep", type=float, default=0.0, help="Extra sleep after publishing one chunk.")
    parser.add_argument(
        "--enable_model_resize",
        action="store_true",
        help="Keep framework-internal resize. By default, resize is controlled in ros_action_to_aloha.",
    )
    parser.add_argument(
        "--use_checkpoint_stats",
        action="store_true",
        help="Deprecated no-op. ResVLA checkpoints always load dataset statistics from the checkpoint run directory.",
    )
    parser.add_argument("--unnorm_key", type=str, default=None, help="Dataset statistics key for action unnormalization.")
    parser.add_argument(
        "--action_scale",
        type=parse_float_list,
        default=parse_float_list("1,1,1,1,1,1,14.3717,1,1,1,1,1,1,14.4678"),
        help="Comma-separated scale used to restore actions.",
    )
    parser.add_argument(
        "--use_state",
        action="store_true",
        help="Flag to enable state usage"
    )
    parser.add_argument(
        "--state_scale",
        type=parse_float_list,
        default=parse_float_list("1,1,1,1,1,1,0.4305,1,1,1,1,1,1,0.4218"),
        help="Comma-separated scale used to parse state when parsing state from payload.",
    )
    parser.add_argument(
        "--use_delta",
        action="store_true",
        help="Decode each predicted chunk as delta actions relative to the chunk-start state.",
    )
    parser.add_argument("--obs_topic", type=str, default="/resvla/observation")
    parser.add_argument("--pred_action_topic", type=str, default="/resvla/predicted_action")
    parser.add_argument("--pred_action_chunk_topic", type=str, default="/resvla/predicted_action_chunk")

    args = parser.parse_args()
    if isinstance(args.policy_cameras, str):
        args.policy_cameras = [item.strip() for item in args.policy_cameras.split(",") if item.strip()]
    
    if args.use_delta:
        args.use_state = True
    return args


if __name__ == "__main__":
    server = ResVLAPolicyServer(get_arguments())
    server.run()

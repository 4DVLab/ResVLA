
# Optional: print more diagnostics for debugging
# from sapien import disable_renderer
# disable_renderer()  # <-- Uncomment to skip the renderer



from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.utils.visualization import write_video
from mani_skill2_real2sim import ASSET_DIR as MS2_REAL2SIM_ASSET_DIR
from pathlib import Path
import logging

logging.basicConfig(level=logging.DEBUG)

env_name = "PutEggplantInBasketScene-v0"

kwargs = {
    "obs_mode": "rgbd",
    "robot": "widowx_sink_camera_setup",
    "sim_freq": 500,
    "control_mode": "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
    "control_freq": 5,
    "max_episode_steps": 120,
    "scene_name": "bridge_table_1_v2",
    "camera_cfgs": {"add_segmentation": True},
    # Headless-friendly renderer init (avoids requiring an X11 DISPLAY in many setups)
    "renderer_kwargs": {"offscreen_only": True},
    # Use an absolute path derived from the installed ManiSkill2_real2sim package.
    # This avoids dependency on current working directory.
    "rgb_overlay_path": str(Path(MS2_REAL2SIM_ASSET_DIR) / "real_inpainting/bridge_sink.png")
}

additional_env_build_kwargs = {}

print("🔧 Start building ManiSkill2 env...")
env = build_maniskill2_env(
    env_name,
    **additional_env_build_kwargs,
    **kwargs,
)
print("✅ Env built successfully:", env)

obs = env.reset()
print("📷 First observation keys:", obs.keys() if isinstance(obs, dict) else type(obs))

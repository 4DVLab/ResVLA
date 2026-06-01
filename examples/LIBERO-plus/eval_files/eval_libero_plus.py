import dataclasses
import datetime as dt
import json
import logging
import math
import os
import pathlib
from pathlib import Path
import requests
import time
import sys
import re
import torch

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.benchmark import grab_language_from_filename
from libero.libero.envs import OffScreenRenderEnv
from collections import defaultdict
os.environ["TOKENIZERS_PARALLELISM"] = "false"

THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR))
from model2libero_interface import ModelClient


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093
    resize_size = [224,224]

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90, all
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 1  # LIBERO-plus recommends 1 rollout per task
    category_filter: str = "all"  # Comma-separated: camera,robot,language,light,background,noise,layout,clean,all

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "results/libero_plus/videos"  # Path to save videos
    log_out_path: str = "results/libero_plus/logs"  # Path to save summary/log artifacts
    resume: bool = False  # Resume from existing episode_log_path if present

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = ""

    post_process_action: bool = True

    job_name: str = "test"


def eval_libero(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    # Set random seed
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    suite_names = [
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
        "libero_90",
    ]
    if args.task_suite_name != "all":
        if args.task_suite_name not in suite_names:
            raise ValueError(f"Unknown task suite: {args.task_suite_name}")
        suite_names = [args.task_suite_name]

    category_filter = _parse_category_filter(args.category_filter)
    classification = _load_task_classification()

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.log_out_path).mkdir(parents=True, exist_ok=True)

    client_model = ModelClient(
        policy_ckpt_path=args.pretrained_path, # to get unnormalization stats
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
    )

    category_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"success": 0, "total": 0})

    ckpt_step_name = _get_checkpoint_step_name(args.pretrained_path)
    episode_log_name = f"episodes_{ckpt_step_name}.jsonl" if ckpt_step_name else "episodes.json"
    summary_log_name = f"libero_plus_summary_{ckpt_step_name}.json" if ckpt_step_name else "libero_plus_summary.json"
    episode_log_path = str(pathlib.Path(args.log_out_path) / episode_log_name)
    logging.info(f"Episode log path: {episode_log_path}")
    completed = set()
    total_episodes = 0
    total_successes = 0
    if args.resume and os.path.exists(episode_log_path):
        completed, category_stats, total_episodes, total_successes = _load_existing_episode_log(
            episode_log_path,
            set(suite_names),
        )
    else:
        with open(episode_log_path, "w", encoding="utf-8") as f:
            f.write("")  # create/overwrite existing file

    # Start evaluation
    for suite_name in suite_names:
        task_suite = benchmark_dict[suite_name]()
        num_tasks_in_suite = task_suite.n_tasks
        logging.info(f"Task suite: {suite_name} ({num_tasks_in_suite} tasks)")

        if suite_name == "libero_spatial":
            max_steps = 220  # longest training demo has 193 steps
        elif suite_name == "libero_object":
            max_steps = 280  # longest training demo has 254 steps
        elif suite_name == "libero_goal":
            max_steps = 300  # longest training demo has 270 steps
        elif suite_name == "libero_10":
            max_steps = 520  # longest training demo has 505 steps
        elif suite_name == "libero_90":
            max_steps = 400  # longest training demo has 373 steps
        else:
            raise ValueError(f"Unknown task suite: {suite_name}")

        suite_video_out = pathlib.Path(args.video_out_path) / suite_name
        suite_video_out.mkdir(parents=True, exist_ok=True)

        task_episodes, task_successes = 0, 0
        suite_entries = classification["entries"].get(suite_name, [])
        if not suite_entries:
            logging.warning(f"No classification entries found for suite: {suite_name}")
            continue

        for entry in tqdm.tqdm(suite_entries):
            task_name = entry.get("name")
            task_category = entry.get("category", "Unclassified")
            if not task_name:
                continue
            if category_filter is not None and task_category.lower() not in category_filter:
                continue
            episode_key = f"{suite_name}::{task_name}::{0}"
            if args.resume and episode_key in completed:
                continue

            initial_states = _load_init_states_for_entry(task_name, suite_name)
            print(suite_name, task_name)
            env, task_description = _get_libero_env_from_bddl(
                suite_name,
                task_name,
                LIBERO_ENV_RESOLUTION,
                args.seed,
            )

            for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
                logging.info(f"\nTask: {task_description}")

                client_model.reset(task_description=task_description)  # Reset the client connection
                env.reset()

                obs = env.set_init_state(initial_states[episode_idx])

                t = 0
                # replay_images = []
                full_actions = []

                logging.info(f"Starting episode {task_episodes + 1}...")
                step = 0
                
                while t < max_steps + args.num_steps_wait:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(
                        obs["robot0_eye_in_hand_image"][::-1, ::-1]
                    )

                    # replay_images.append(img)

                    state = np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    )

                    observation = {
                        "observation.primary": np.expand_dims(
                            img, axis=0
                        ),
                        "observation.wrist_image": np.expand_dims(
                            wrist_img, axis=0
                        ),
                        "observation.state": np.expand_dims(state, axis=0),
                        "instruction": [str(task_description)],
                    }

                    example_dict = {
                        "image": [observation["observation.primary"][0], observation["observation.wrist_image"][0]],
                        "lang": observation["instruction"][0],
                    }

                    response = client_model.step(example=example_dict, step=step) 

                    raw_action = response["raw_action"]
                    
                    world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                    rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                    open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                    gripper = _binarize_gripper_open(open_gripper)

                    if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                        logging.warning(f"Unexpected action sizes: "
                                        f"wv={world_vector_delta.shape}, rot={rotation_delta.shape}, grip={gripper.shape}. "
                                        f"Falling back to LIBERO_DUMMY_ACTION.")
                        raise ValueError(
                            f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                            f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                        )
                    else:
                        delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)

                    full_actions.append(delta_action)
                    
                    obs, reward, done, info = env.step(delta_action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                    step += 1

                task_episodes += 1
                total_episodes += 1

                category_stats[task_category]["total"] += 1
                if done:
                    category_stats[task_category]["success"] += 1

                total_success_rate = (float(total_successes) / float(total_episodes)) if total_episodes else None
                category_total = int(category_stats[task_category]["total"])
                category_successes = int(category_stats[task_category]["success"])
                category_success_rate = (float(category_successes) / float(category_total)) if category_total else None

                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                # imageio.mimwrite(
                #     suite_video_out
                #     / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4",
                #     [np.asarray(x) for x in replay_images],
                #     fps=10,
                # )

                _append_episode_log(
                    episode_log_path,
                    {
                        "suite": suite_name,
                        "task_name": task_name,
                        "task_category": task_category,
                        "episode_idx": episode_idx,
                        "success": bool(done),
                        "steps": int(step),
                        "seed": args.seed,
                        "timestamp": dt.datetime.now().isoformat(),
                        "total_episodes_so_far": int(total_episodes),
                        "total_successes_so_far": int(total_successes),
                        "total_success_rate": total_success_rate,
                        "category_total_so_far": category_total,
                        "category_successes_so_far": category_successes,
                        "category_success_rate": category_success_rate,
                    },
                )
                completed.add(episode_key)

                logging.info(f"Success: {done}")
                logging.info(f"# episodes completed so far: {total_episodes}")
                logging.info(f"# successes so far: {total_successes}")

    summary = {
        key: {
            "success": value["success"],
            "total": value["total"],
            "success_rate": (value["success"] / value["total"]) if value["total"] else None,
        }
        for key, value in category_stats.items()
    }
    logging.info("LIBERO-plus category summary:")
    for key, value in summary.items():
        if value["total"]:
            logging.info(f"  {key}: {value['success']}/{value['total']}")

    summary_path = pathlib.Path(args.log_out_path) / summary_log_name
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _get_libero_env_from_bddl(task_suite: str, task_name: str, resolution: int, seed: int):
    """Initializes the LIBERO environment from a bddl entry name."""
    bddl_file = f"{task_name}.bddl"
    bddl_file_path = os.path.join(
        get_libero_path("bddl_files"),
        task_suite,
        bddl_file,
    )
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file_path,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)

    task_description = grab_language_from_filename(task_suite, bddl_file)
    return env, task_description


def _load_task_classification() -> dict[str, dict[str, object]]:
    classification_path = os.path.join(get_libero_path("assets"), "../benchmark/task_classification.json")
    with open(classification_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping: dict[str, dict[str, object]] = {"entries": {}, "category_by_name": {}}
    for suite_name, tasks in data.items():
        suite_map: dict[str, str] = {}
        for entry in tasks:
            task_name = entry.get("name")
            category = entry.get("category", "Unclassified")
            if task_name:
                suite_map[task_name] = category
        mapping["entries"][suite_name] = tasks
        mapping["category_by_name"][suite_name] = suite_map
    return mapping


def _parse_category_filter(category_filter: str) -> set[str] | None:
    value = (category_filter or "all").strip().lower()
    if value == "all":
        return None
    return {x.strip().lower() for x in value.split(",") if x.strip()}


def _get_checkpoint_step_name(pretrained_path: str) -> str | None:
    match = re.search(r"steps[_-](\d+)", str(pretrained_path or ""))
    if match is None:
        return None
    return f"steps_{match.group(1)}"


def _append_episode_log(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_existing_episode_log(
    path: str,
    suite_filter: set[str],
) -> tuple[set[str], dict[str, dict[str, int]], int, int]:
    completed: set[str] = set()
    category_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"success": 0, "total": 0})
    total_episodes = 0
    total_successes = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            suite = record.get("suite")
            if suite_filter and suite not in suite_filter:
                continue
            task_name = record.get("task_name")
            episode_idx = record.get("episode_idx")
            if task_name is None or episode_idx is None:
                continue
            key = f"{suite}::{task_name}::{episode_idx}"
            completed.add(key)
            total_episodes += 1
            if record.get("success"):
                total_successes += 1
            category = record.get("task_category", "Unclassified")
            category_stats[category]["total"] += 1
            if record.get("success"):
                category_stats[category]["success"] += 1
    return completed, category_stats, total_episodes, total_successes


def _load_init_states_for_entry(task_name: str, task_suite: str):
    init_states_path, is_newobj = _resolve_init_states_path(task_name, task_suite)
    init_states = torch.load(init_states_path, weights_only=False)
    if is_newobj:
        init_states = init_states.reshape(1, -1)
    return init_states


def _resolve_init_states_path(task_name: str, task_suite: str) -> tuple[str, bool]:
    init_root = get_libero_path("init_states")

    base_name = task_name
    if "_language_" in base_name:
        base_name = base_name.split("_language_")[0]
    if "_view_" in base_name:
        base_name = base_name.split("_view_")[0]
    if "_light_" in base_name:
        base_name = base_name.split("_light_")[0]

    base_name = re.sub(r"_table_\d+", "", base_name)
    base_name = re.sub(r"_tb_\d+", "", base_name)

    candidates = [
        task_name,
        base_name,
        re.sub(r"_add_\d+", "", base_name),
        re.sub(r"_level\d+", "", base_name),
    ]

    candidates = [c for c in candidates if c]
    init_files = [f"{c}.pruned_init" for c in candidates]
    not_found = []

    for init_file in init_files:
        candidate_path = os.path.join(init_root, "libero_newobj", task_suite, init_file)
        if os.path.exists(candidate_path):
            return candidate_path, True
        
        not_found.append(candidate_path)
    
    for init_file in init_files:
        candidate_path = os.path.join(init_root, task_suite, init_file)
        if os.path.exists(candidate_path):
            return candidate_path, False

    raise FileNotFoundError(
        f"Init states file not found for task '{task_name}' in suite '{task_suite}'. "
        f"Tried: {', '.join(not_found)}"
    )


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert a quaternion to axis-angle format."""
    # normalize
    quat = quat / np.linalg.norm(quat)
    # angle = 2 * acos(w)
    angle = 2 * math.acos(quat[0])
    # axis = v / sin(angle/2)
    axis = quat[1:] / math.sin(angle / 2)
    # axis-angle
    return axis * angle


if __name__ == "__main__":
    args = tyro.cli(Args)
    eval_libero(args)

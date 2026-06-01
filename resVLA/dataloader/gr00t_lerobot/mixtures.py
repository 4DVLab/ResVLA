"""
mixtures.py

Defines a registry of dataset mixtures and weights for the Open-X Embodiment Datasets. Each dataset is associated with
a float "sampling weight"
"""

from typing import Dict, List, Tuple


# Dataset mixture name mapped to a list of tuples containing:
## {nakename: [(data_name, sampling_weight, robot_type)] }
DATASET_NAMED_MIXTURES = {

    "custom_dataset": [
        ("custom_dataset_name", 1.0, "custom_robot_config"),
    ],
    "custom_dataset_2": [
        ("custom_dataset_name_1", 1.0, "custom_robot_config"),
        ("custom_dataset_name_2", 1.0, "custom_robot_config"),
    ],

    "libero_all": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
                # ("libero_90_no_noops_lerobot", 1.0, "libero_franka"),
    ],
    "libero_plus": [
        ("libero_plus_lerobot", 1.0, "libero_franka"),
    ],
    "libero_all_with_plus": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_plus_lerobot", 1.0, "libero_franka"),
                # ("libero_90_no_noops_lerobot", 1.0, "libero_franka"),
    ],
    "simpler_env_bridge": [
        ("bridge_orig_1.0.0_lerobot", 1.0, "oxe_bridge"),
    ],
    "simpler_env_fractal": [
        ("fractal20220817_data_0.1.0_lerobot", 1.0, "oxe_rt1"),
    ],
    "simpler_env_all": [
        ("bridge_orig_1.0.0_lerobot", 1.0, "oxe_bridge"),
        ("fractal20220817_data_0.1.0_lerobot", 1.0, "oxe_rt1"),
    ],

    "demo_sim_pick_place": [
        ("sim_pick_place", 1.0, "demo_sim_franka_delta_joints"),
    ],

    "custom_dataset": [
        ("custom_dataset_name", 1.0, "custom_robot_config"),
    ],
    "custom_dataset_2": [
        ("custom_dataset_name_1", 1.0, "custom_robot_config"),
        ("custom_dataset_name_2", 1.0, "custom_robot_config"),
    ],

    "fourier_gr1_unified_1000": [
        ("gr1_unified.PnPBottleToCabinetClose", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPCanToDrawerClose", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPCupToDrawerClose", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPMilkToMicrowaveClose", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPPotatoToMicrowaveClose", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPWineToCabinetClose", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA", 1.0, "fourier_gr1_arms_waist"),
    ],

    "BEHAVIOR_challenge": [
        ("BEHAVIOR_challenge", 1.0, "R1Pro"),
    ],


    "SO101_pick": [
        ("pick_dataset_name", 1.0, "SO101"),
    ],

    "arx_x5": [
        ("arx_x5", 1.0, "arx_x5"),
    ],

    "robotwin": [
        ("adjust_bottle", 1.0, "robotwin"),
        ("beat_block_hammer", 1.0, "robotwin"),
        ("blocks_ranking_rgb", 1.0, "robotwin"),
        ("blocks_ranking_size", 1.0, "robotwin"),
        ("click_alarmclock", 1.0, "robotwin"),
        ("click_bell", 1.0, "robotwin"),
        ("dump_bin_bigbin", 1.0, "robotwin"),
        ("grab_roller", 1.0, "robotwin"),
        ("handover_block", 1.0, "robotwin"),
        ("handover_mic", 1.0, "robotwin"),
        ("hanging_mug", 1.0, "robotwin"),
        ("lift_pot", 1.0, "robotwin"),
        ("move_can_pot", 1.0, "robotwin"),
        ("move_pillbottle_pad", 1.0, "robotwin"),
        ("move_playingcard_away", 1.0, "robotwin"),
        ("move_stapler_pad", 1.0, "robotwin"),
        ("open_laptop", 1.0, "robotwin"),
        ("open_microwave", 1.0, "robotwin"),
        ("pick_diverse_bottles", 1.0, "robotwin"),
        ("pick_dual_bottles", 1.0, "robotwin"),
        ("place_a2b_left", 1.0, "robotwin"),
        ("place_a2b_right", 1.0, "robotwin"),
        ("place_bread_basket", 1.0, "robotwin"),
        ("place_bread_skillet", 1.0, "robotwin"),
        ("place_burger_fries", 1.0, "robotwin"),
        ("place_can_basket", 1.0, "robotwin"),
        ("place_cans_plasticbox", 1.0, "robotwin"),
        ("place_container_plate", 1.0, "robotwin"),
        ("place_dual_shoes", 1.0, "robotwin"),
        ("place_empty_cup", 1.0, "robotwin"),
        ("place_fan", 1.0, "robotwin"),
        ("place_mouse_pad", 1.0, "robotwin"),
        ("place_object_basket", 1.0, "robotwin"),
        ("place_object_scale", 1.0, "robotwin"),
        ("place_object_stand", 1.0, "robotwin"),
        ("place_phone_stand", 1.0, "robotwin"),
        ("place_shoe", 1.0, "robotwin"),
        ("press_stapler", 1.0, "robotwin"),
        ("put_bottles_dustbin", 1.0, "robotwin"),
        ("put_object_cabinet", 1.0, "robotwin"),
        ("rotate_qrcode", 1.0, "robotwin"),
        ("scan_object", 1.0, "robotwin"),
        ("shake_bottle", 1.0, "robotwin"),
        ("shake_bottle_horizontally", 1.0, "robotwin"),
        ("stack_blocks_three", 1.0, "robotwin"),
        ("stack_blocks_two", 1.0, "robotwin"),
        ("stack_bowls_three", 1.0, "robotwin"),
        ("stack_bowls_two", 1.0, "robotwin"),
        ("stamp_seal", 1.0, "robotwin"),
        ("turn_switch", 1.0, "robotwin"),
    ],

    "robotwin_task1": [
        ("adjust_bottle", 1.0, "robotwin"),
    ],
    "robotwin_task2": [
        ("place_a2b_left", 1.0, "robotwin"),
        ("place_a2b_right", 1.0, "robotwin"),
    ],

    "multi_robot": [
        ("LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        # ("OXE_LEROBOT_DATASET/bridge_orig_1.0.0_lerobot", 1.0, "oxe_bridge"),
    ],
    "dual_arm": [
        ("dual_arm_tasks/flower_vase_v2", 1.0, "aloha"),
        ("dual_arm_tasks/handover_package", 1.0, "aloha"),
        ("dual_arm_tasks/pen_in_container", 1.0, "aloha"),
        ("dual_arm_tasks/pot_top_v2", 1.0, "aloha"),
    ],

    "single_arm": [
        ("single_arm_tasks/charger_unplugin", 1.0, "aloha"),
        ("single_arm_tasks/open_box", 1.0, "aloha"),
        ("single_arm_tasks/sweep_trash", 1.0, "aloha"),
        ("single_arm_tasks/sweep_whiteboard", 1.0, "aloha"),
    ],
    
    "aloha_demo": [
        ("dual_arm_tasks/cup_shelf", 1.0, "aloha"),
    ]
}

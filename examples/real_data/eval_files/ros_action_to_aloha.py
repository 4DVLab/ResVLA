import argparse
import base64
import json
import threading
from collections import deque

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Header, String


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def encode_image_to_b64(image: np.ndarray, jpeg_quality: int) -> str:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise ValueError("Failed to encode image to JPEG.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def resize_for_policy(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if width <= 0 or height <= 0:
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


class AlohaRosBridge:
    def __init__(self, args):
        self.args = args
        self.bridge = CvBridge()

        self.img_front_deque = deque()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_high_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()

        self.obs_publisher = None
        self.puppet_arm_left_publisher = None
        self.puppet_arm_right_publisher = None

        self.publish_thread = None
        self.publish_lock = threading.Lock()
        self.publish_lock.acquire()

        self.init_ros()

    def init_ros(self):
        rospy.init_node("resvla_action_bridge", anonymous=True)

        image_sub_args = {
            "queue_size": 20,
            "buff_size": 65536 * 8,
            "tcp_nodelay": False,
        }
        joint_sub_args = {
            "queue_size": 1000,
            "tcp_nodelay": True,
        }

        rospy.Subscriber(self.args.img_left_topic, Image, self.img_left_callback, **image_sub_args)
        rospy.Subscriber(self.args.img_right_topic, Image, self.img_right_callback, **image_sub_args)
        rospy.Subscriber(self.args.img_front_topic, Image, self.img_front_callback, **image_sub_args)
        rospy.Subscriber(self.args.img_high_topic, CompressedImage, self.img_high_callback, queue_size=10, buff_size=65536 * 8)
        rospy.Subscriber(self.args.puppet_arm_left_topic, JointState, self.puppet_arm_left_callback, **joint_sub_args)
        rospy.Subscriber(self.args.puppet_arm_right_topic, JointState, self.puppet_arm_right_callback, **joint_sub_args)

        # Keep only the newest action to avoid delayed execution from stale backlog.
        rospy.Subscriber(self.args.pred_action_topic, JointState, self.action_callback, queue_size=1, tcp_nodelay=True)

        self.obs_publisher = rospy.Publisher(self.args.obs_topic, String, queue_size=4)
        self.puppet_arm_left_publisher = rospy.Publisher(self.args.puppet_arm_left_cmd_topic, JointState, queue_size=10)
        self.puppet_arm_right_publisher = rospy.Publisher(self.args.puppet_arm_right_cmd_topic, JointState, queue_size=10)

    def _append_with_limit(self, dq: deque, msg, max_len: int = 2000):
        if len(dq) >= max_len:
            dq.popleft()
        dq.append(msg)

    def img_left_callback(self, msg):
        self._append_with_limit(self.img_left_deque, msg)

    def img_right_callback(self, msg):
        self._append_with_limit(self.img_right_deque, msg)

    def img_front_callback(self, msg):
        self._append_with_limit(self.img_front_deque, msg)

    def img_high_callback(self, msg):
        self._append_with_limit(self.img_high_deque, msg)

    def puppet_arm_left_callback(self, msg):
        self._append_with_limit(self.puppet_arm_left_deque, msg)

    def puppet_arm_right_callback(self, msg):
        self._append_with_limit(self.puppet_arm_right_deque, msg)

    def get_frame(self):
        if (
            len(self.img_left_deque) == 0
            or len(self.img_right_deque) == 0
            or len(self.img_front_deque) == 0
            or len(self.img_high_deque) == 0
            or len(self.puppet_arm_left_deque) == 0
            or len(self.puppet_arm_right_deque) == 0
        ):
            return None

        frame_time = min(
            [
                self.img_left_deque[-1].header.stamp.to_sec(),
                self.img_right_deque[-1].header.stamp.to_sec(),
                self.img_front_deque[-1].header.stamp.to_sec(),
                self.img_high_deque[-1].header.stamp.to_sec(),
                self.puppet_arm_left_deque[-1].header.stamp.to_sec(),
                self.puppet_arm_right_deque[-1].header.stamp.to_sec(),
            ]
        )

        for dq in [
            self.img_left_deque,
            self.img_right_deque,
            self.img_front_deque,
            self.img_high_deque,
            self.puppet_arm_left_deque,
            self.puppet_arm_right_deque,
        ]:
            if len(dq) == 0 or dq[-1].header.stamp.to_sec() < frame_time:
                return None

        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), "passthrough")

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), "passthrough")

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), "passthrough")

        while self.img_high_deque[0].header.stamp.to_sec() < frame_time:
            self.img_high_deque.popleft()
        img_high = self.bridge.compressed_imgmsg_to_cv2(self.img_high_deque.popleft(), "passthrough")

        while self.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_left_deque.popleft()
        puppet_arm_left = self.puppet_arm_left_deque.popleft()

        while self.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_right_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()

        return img_front, img_high, img_left, img_right, puppet_arm_left, puppet_arm_right

    def build_observation_payload(self, frame) -> str:
        img_front, img_high, img_left, img_right, puppet_arm_left, puppet_arm_right = frame
        qpos = np.concatenate(
            (np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)),
            axis=0,
        ).astype(np.float32)

        img_front = resize_for_policy(img_front, self.args.image_width, self.args.image_height)
        img_high = resize_for_policy(img_high, self.args.image_width, self.args.image_height)
        img_left = resize_for_policy(img_left, self.args.image_width, self.args.image_height)
        img_right = resize_for_policy(img_right, self.args.image_width, self.args.image_height)

        payload = {
            "prompt": self.args.prompt,
            "state": qpos.tolist(),
            "images": {
                "cam_front": encode_image_to_b64(img_front, self.args.jpeg_quality),
                "cam_high": encode_image_to_b64(img_high, self.args.jpeg_quality),
                "cam_left": encode_image_to_b64(img_left, self.args.jpeg_quality),
                "cam_right": encode_image_to_b64(img_right, self.args.jpeg_quality),
            },
        }
        return json.dumps(payload, separators=(",", ":"))

    def current_qpos(self):
        if len(self.puppet_arm_left_deque) == 0 or len(self.puppet_arm_right_deque) == 0:
            return None, None
        left_arm = np.asarray(self.puppet_arm_left_deque[-1].position, dtype=np.float32)
        right_arm = np.asarray(self.puppet_arm_right_deque[-1].position, dtype=np.float32)
        return left_arm, right_arm

    def build_joint_state(self, position):
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()
        joint_state_msg.name = [f"joint{i}" for i in range(len(position))]
        joint_state_msg.position = position.tolist()
        return joint_state_msg

    def publish_once(self, left, right):
        self.puppet_arm_left_publisher.publish(self.build_joint_state(left))
        self.puppet_arm_right_publisher.publish(self.build_joint_state(right))

    def publish_continuous(self, left_target, right_target):
        rate = rospy.Rate(self.args.publish_rate)
        current_left, current_right = None, None
        while not rospy.is_shutdown():
            current_left, current_right = self.current_qpos()
            if current_left is not None and current_right is not None:
                break
            rate.sleep()

        left_symbol = np.where(left_target - current_left > 0, 1.0, -1.0)
        right_symbol = np.where(right_target - current_right > 0, 1.0, -1.0)
        left_gripper = left_target[-1]
        right_gripper = right_target[-1]

        while not rospy.is_shutdown():
            if self.publish_lock.acquire(False):
                return

            left_diff = np.abs(left_target - current_left)
            right_diff = np.abs(right_target - current_right)
            finished = True

            for idx, diff in enumerate(left_diff):
                if diff < self.args.arm_steps_length[idx]:
                    current_left[idx] = left_target[idx]
                else:
                    current_left[idx] += left_symbol[idx] * self.args.arm_steps_length[idx]
                    finished = False

            for idx, diff in enumerate(right_diff):
                if diff < self.args.arm_steps_length[idx]:
                    current_right[idx] = right_target[idx]
                else:
                    current_right[idx] += right_symbol[idx] * self.args.arm_steps_length[idx]
                    finished = False

            current_left[-1] = left_gripper
            current_right[-1] = right_gripper
            self.publish_once(current_left, current_right)
            if finished:
                return
            rate.sleep()

    def publish_linear(self, left_target, right_target):
        current_left, current_right = None, None
        rate = rospy.Rate(self.args.publish_rate)
        while not rospy.is_shutdown():
            current_left, current_right = self.current_qpos()
            if current_left is not None and current_right is not None:
                break
            rate.sleep()

        left_traj = np.linspace(current_left, left_target, self.args.linear_interp_steps)
        right_traj = np.linspace(current_right, right_target, self.args.linear_interp_steps)
        for left_step, right_step in zip(left_traj, right_traj):
            if self.publish_lock.acquire(False):
                return
            left_step[-1] = left_target[-1]
            right_step[-1] = right_target[-1]
            self.publish_once(left_step, right_step)
            rate.sleep()

    def publish_threaded(self, left_target, right_target):
        if self.publish_thread is not None:
            self.publish_lock.release()
            self.publish_thread.join()
            self.publish_lock.acquire(False)
            self.publish_thread = None

        target_fn = self.publish_linear if self.args.interpolation == "linear" else self.publish_continuous
        self.publish_thread = threading.Thread(target=target_fn, args=(left_target, right_target))
        self.publish_thread.daemon = True
        self.publish_thread.start()

    def action_callback(self, msg):
        action = np.asarray(msg.position, dtype=np.float32)
        if action.shape[0] != 14:
            rospy.logwarn_throttle(2.0, f"Expected 14-dim action, got {action.shape[0]}")
            return

        left_target = action[:7]
        right_target = action[7:14]
        if self.args.execute_as_delta:
            current_left, current_right = self.current_qpos()
            if current_left is None or current_right is None:
                return
            left_target = current_left + left_target
            right_target = current_right + right_target

        if self.args.action_execute_mode == "direct":
            self.publish_once(left_target, right_target)
        else:
            self.publish_threaded(left_target, right_target)

    def home_robot_on_start(self):
        if self.args.skip_home_on_start:
            rospy.loginfo("Skip homing on start as requested.")
            return

        home_left = np.asarray(self.args.home_left, dtype=np.float32)
        home_right = np.asarray(self.args.home_right, dtype=np.float32)
        if home_left.shape[0] != 7 or home_right.shape[0] != 7:
            rospy.logwarn("Homing target must be 7 DoF for each arm. Skip homing.")
            return

        rospy.loginfo("Waiting for arm joint states before homing...")
        wait_rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            current_left, current_right = self.current_qpos()
            if current_left is not None and current_right is not None:
                break
            wait_rate.sleep()

        if rospy.is_shutdown():
            return

        rospy.loginfo(f"Homing Aloha arms to init pose: left={home_left.tolist()}, right={home_right.tolist()}")
        self.publish_continuous(home_left, home_right)
        rospy.loginfo("Homing completed.")

    def run(self):
        self.home_robot_on_start()
        rate = rospy.Rate(self.args.control_hz)
        rospy.loginfo(
            f"Aloha ROS bridge is ready. mode={self.args.action_execute_mode}, "
            "publishing observations and listening for actions..."
        )
        while not rospy.is_shutdown():
            frame = self.get_frame()
            if frame is None:
                rospy.loginfo_throttle(5.0, "Waiting for synchronized sensor topics for observation publish...")
                rate.sleep()
                continue

            try:
                payload = self.build_observation_payload(frame)
                msg = String()
                msg.data = payload
                self.obs_publisher.publish(msg)
            except Exception as exc:
                rospy.logerr_throttle(5.0, f"Observation publish error, keep running: {exc}")

            rate.sleep()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Hang the teacup on the cup rack.")
    parser.add_argument("--obs_topic", type=str, default="/resvla/observation")
    parser.add_argument("--pred_action_topic", type=str, default="/resvla/predicted_action")
    parser.add_argument("--puppet_arm_left_cmd_topic", type=str, default="/master/joint_left")
    parser.add_argument("--puppet_arm_right_cmd_topic", type=str, default="/master/joint_right")
    parser.add_argument("--puppet_arm_left_topic", type=str, default="/puppet/joint_left")
    parser.add_argument("--puppet_arm_right_topic", type=str, default="/puppet/joint_right")
    parser.add_argument("--img_front_topic", type=str, default="/camera_f/color/image_raw")
    parser.add_argument("--img_left_topic", type=str, default="/camera_l/color/image_raw")
    parser.add_argument("--img_right_topic", type=str, default="/camera_r/color/image_raw")
    parser.add_argument("--img_high_topic", type=str, default="/rgb/image_raw/compressed")
    parser.add_argument("--control_hz", type=int, default=10)
    parser.add_argument("--publish_rate", type=int, default=30)
    parser.add_argument("--image_width", type=int, default=640, help="Resize width for all policy images. <=0 disables resize.")
    parser.add_argument("--image_height", type=int, default=480, help="Resize height for all policy images. <=0 disables resize.")
    parser.add_argument("--jpeg_quality", type=int, default=80)
    parser.add_argument("--linear_interp_steps", type=int, default=100)
    parser.add_argument("--interpolation", type=str, choices=["linear", "continuous"], default="continuous")
    parser.add_argument(
        "--action_execute_mode",
        type=str,
        choices=["direct", "continuous", "linear"],
        default="direct",
        help="Action execution mode. Default `direct` matches inference_pi0 main-loop behavior.",
    )
    parser.add_argument("--execute_as_delta", action="store_true", help="Interpret the received action as delta-qpos.")
    parser.add_argument("--skip_home_on_start", action="store_true", help="Skip moving robot to initial pose on startup.")
    parser.add_argument(
        "--home_left",
        type=parse_float_list,
        default="0.00629425,0.01049042,0.02651215,-0.14171791,0.04558659,-0.17833996,-0.01332873",
        help="Comma-separated 7-DoF startup home pose for left arm.",
    )
    parser.add_argument(
        "--home_right",
        type=parse_float_list,
        default="-0.03490543,0.01316071,0.02422333,-0.1993208,-0.19893932,-0.03337955,-0.02008527",
        help="Comma-separated 7-DoF startup home pose for right arm.",
    )
    parser.add_argument(
        "--arm_steps_length",
        type=parse_float_list,
        default="0.005,0.005,0.005,0.005,0.005,0.005,0.2",
        help="Comma-separated interpolation step for each arm joint.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    bridge = AlohaRosBridge(get_arguments())
    if bridge.args.action_execute_mode in ("continuous", "linear"):
        bridge.args.interpolation = bridge.args.action_execute_mode
    bridge.run()

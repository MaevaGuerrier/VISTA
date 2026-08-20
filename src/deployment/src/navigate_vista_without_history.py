import argparse
import gc
import os
import pprint
import time

import cv2
import numpy as np

# ROS2
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, PoseStamped, Twist
from nav_msgs.msg import Path
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Float32MultiArray, Int32
from topic_names import (
    CLOSEST_NODE_TOPIC,
    IMAGE_TOPIC,
    SAMPLED_ACTIONS_TOPIC,
    WAYPOINT_TOPIC,
)
from utils import pil_to_numpy_array, plot_trajs_and_points_on_image
from utils_onnx import load_model_onnx, msg_to_pil, transform_images

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WORK_DIR = "/workspace/src/deployment/"
TOPOMAP_IMAGES_DIR = f"{WORK_DIR}topomaps/images"
MODEL_WEIGHTS_PATH = f"{WORK_DIR}model_weights/"
ROBOT_CONFIG_PATH = f"{WORK_DIR}config/robot.yaml"
MODEL_CONFIG_PATH = f"{WORK_DIR}../train/config/"

with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)

MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"]
DT = 1.0 / RATE
EPS = 1e-8
VEL_TOPIC = robot_config["vel_navi_topic"]

MODEL_PARAMS = {
    "normalize": True,
    "context_size": 5,
    "image_size": [192, 192],
}

# Camera calibration
INTRINSICS = np.array(
    [
        [235.7444344725863, 2.2822917369575983, 320.3212422370101],
        [0.0, 237.67070839912813, 232.78147845844464],
        [0.0, 0.0, 1.0],
    ]
)
CAMERA_HEIGHT = 0.560
CAMERA_X_OFFSET = 0.200

EXTRINSICS = np.array(
    [
        [0, 0, 1, -CAMERA_X_OFFSET],
        [-1, 0, 0, -0.000],
        [0, -1, 0, -CAMERA_HEIGHT],
        [0, 0, 0, 1],
    ]
)

DIST_COEFF = np.array(
    [
        [-0.053129475318406234],
        [0.03335273788977895],
        [-0.031760136310879046],
        [0.008394411829175783],
    ]
)

VIZ_IMAGE_SIZE_FISHEYE = (640, 480)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class NavNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("EXPLORATION")
        self.args = args

        # cv2
        self.bridge = CvBridge()

        # ---- State --------------------------------------------------------
        self.context_queue: list = []
        self.action_history: list = []
        self.context_size: int = MODEL_PARAMS["context_size"]
        self.latest_image = None
        self.vel_msg = Twist()
        self.sg_node: int = 0
        self.chosen_waypoint = np.zeros(4)

        # ---- Model --------------------------------------------------------
        self.model = load_model_onnx(f"{args.model}.onnx")
        self.get_logger().info("Loaded ONNX model.")
        active_providers = self.model.get_providers()
        assert "CUDAExecutionProvider" in active_providers, (
            f"GPU requested but not active. Current providers: {active_providers}"
        )
        self.get_logger().info(f"Model running on device: {active_providers[0]}.")

        # ---- Topomap ------------------------------------------------------

        self.topomap, self.goal_node = self._load_topomap(args.dir, args.goal_node)
        num_nodes = len(self.topomap)
        self.get_logger().info(f"Loaded topomap with {num_nodes} nodes.")

        assert -1 <= self.goal_node < num_nodes, "Invalid goal index"
        self.closest_node = 0
        self.reached_goal = False

        # ---- QoS ----------------------------------------------------------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        reliable_qos = QoSProfile(depth=10)
        reliable_qos_1 = QoSProfile(depth=1)

        # ---- Publishers ---------------------------------------------------
        self.vel_pub = self.create_publisher(Twist, VEL_TOPIC, reliable_qos)
        self.waypoint_pub = self.create_publisher(
            Float32MultiArray, WAYPOINT_TOPIC, reliable_qos_1
        )
        self.waypoint_viz_pub = self.create_publisher(
            PoseStamped, "viz_wp", reliable_qos_1
        )
        self.path_viz_pub = self.create_publisher(Path, "viz_path", reliable_qos_1)
        self.sampled_actions_pub = self.create_publisher(
            Float32MultiArray, SAMPLED_ACTIONS_TOPIC, reliable_qos_1
        )
        self.goal_pub = self.create_publisher(
            Bool, "/topoplan/reached_goal", reliable_qos_1
        )
        self.goal_img_pub = self.create_publisher(
            Image, "/topoplan/goal_img", reliable_qos_1
        )
        self.subgoal_img_pub = self.create_publisher(
            Image, "/topoplan/subgoal_img", reliable_qos_1
        )
        self.closest_node_img_pub = self.create_publisher(
            Image, "/topoplan/closest_node_img", reliable_qos_1
        )
        self.closest_node_pub = self.create_publisher(
            Int32, CLOSEST_NODE_TOPIC, reliable_qos
        )
        self.distances_pub = self.create_publisher(
            Float32MultiArray, "/distances", reliable_qos_1
        )
        self.inference_pub = self.create_publisher(
            Float32, "/inference_time", reliable_qos
        )
        self.img_overlay_pub = self.create_publisher(
            Image, "/wps_overlay_img", reliable_qos_1
        )

        # ---- Subscriber ---------------------------------------------------
        self.image_sub = self.create_subscription(
            Image,
            IMAGE_TOPIC,
            self._image_callback,
            sensor_qos,
        )

        # ---- Output dir ---------------------------------------------------
        os.makedirs(args.viz_path, exist_ok=True)

        # ---- Timer (main nav loop) ----------------------------------------
        self.timer = self.create_timer(DT, self._nav_loop)
        self.get_logger().info("NavNode initialised — waiting for images.")

    def _publish_overlay_image(
        self,
        camera_matrix_orig,
        dist_coeffs,
        img: np.ndarray,
        pub,
        trajs: list[np.ndarray],
        viz_img_size: tuple[int, int],
        camera_height: float,
        camera_x_offset: float,
        resize_factor: bool = False,
    ):

        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        # Convert RGB → BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img = plot_trajs_and_points_on_image(
            img=img,
            camera_matrix=camera_matrix_orig,
            dist_coeffs=dist_coeffs,
            list_trajs=trajs,
            viz_img_size=viz_img_size,
            camera_height=camera_height,
            camera_x_offset=camera_x_offset,
            resize_factor=resize_factor,
        )

        try:
            img = np.ascontiguousarray(img)
            ros_img = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            ros_img.header.stamp = self.get_clock().now().to_msg()
            ros_img.header.frame_id = "base_footprint"
            pub.publish(ros_img)
        except Exception as e:
            self.get_logger().error(f"Failed to publish overlay image: {e}")

    def _clip_angle(self, angle):
        return np.mod(angle + np.pi, 2 * np.pi) - np.pi

    def _load_topomap(
        self, dir_path: str, goal_node: int
    ) -> tuple[list[PILImage.Image], int]:
        topomap_filenames = sorted(
            os.listdir(os.path.join(TOPOMAP_IMAGES_DIR, dir_path)),
            key=lambda x: int(x.split(".")[0]),
        )
        topomap_dir = f"{TOPOMAP_IMAGES_DIR}/{dir_path}"
        num_nodes = len(os.listdir(topomap_dir))
        topomap = []
        for i in range(num_nodes):
            image_path = os.path.join(topomap_dir, topomap_filenames[i])
            topomap.append(PILImage.open(image_path))

        assert -1 <= goal_node < len(topomap), "Invalid goal index for the topomap"
        if goal_node == -1:
            goal_node = len(topomap) - 1

        return topomap, goal_node

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------
    def _image_callback(self, msg: Image) -> None:
        self.latest_image = msg_to_pil(msg)

    # -----------------------------------------------------------------------
    # Navigation loop (called at RATE Hz by the timer)
    # -----------------------------------------------------------------------
    def _nav_loop(self) -> None:
        if self.reached_goal:
            return

        args = self.args

        # Accumulate context
        if self.latest_image is not None:
            self.context_queue.append(self.latest_image)
            if len(self.context_queue) > self.context_size + 1:
                self.context_queue.pop(0)

        if len(self.context_queue) > MODEL_PARAMS["context_size"]:
            start = max(self.closest_node - args.radius, 0)
            end = min(self.closest_node + args.radius + 1, self.goal_node)

            distances = []
            waypoints = []

            crop = True

            t0 = time.time()

            # Transform observation once
            transf_obs_img = transform_images(
                self.context_queue, MODEL_PARAMS["image_size"], center_crop=crop
            )

            # Batch all goal images
            goal_imgs = self.topomap[start : end + 1]
            batch_goal_data_np = np.concatenate(
                [
                    transform_images(
                        sg_img, MODEL_PARAMS["image_size"], center_crop=crop
                    )
                    for sg_img in goal_imgs
                ],
                axis=0,
            ).astype("float32")

            num_goals = len(goal_imgs)
            batch_obs_imgs_np = np.tile(transf_obs_img, (num_goals, 1, 1, 1)).astype(
                "float32"
            )

            inputs = {
                "obs_img": batch_obs_imgs_np,
                "goal_img": batch_goal_data_np,
            }

            # To handle garbage collect and double free corruption C error
            try:
                distances, waypoints = self.model.run(None, inputs)
            except Exception as e:
                self.get_logger().error(f"Inference failed: {e}")
            except KeyboardInterrupt:
                # Catching it here prevents the C++ session from being
                # left in an unrecoverable state during the jump to 'finally'
                self.get_logger().info("Inference interrupted by user.")
                raise  # Re-raise to allow the main loop to catch it

            inference_time = time.time() - t0
            self.get_logger().info(f"Inference: {inference_time:.3f}s")
            pprint.pprint(dict(zip(range(start, end + 1), distances)))

            # Publish inference time
            inf_msg = Float32()
            inf_msg.data = float(inference_time)
            self.inference_pub.publish(inf_msg)

            # Publish distances
            dist_msg = Float32MultiArray()
            dist_msg.data = distances.flatten().tolist()
            self.distances_pub.publish(dist_msg)

            # Closest node / subgoal selection
            min_dist_idx = int(np.argmin(distances))
            if distances[min_dist_idx] > args.close_threshold:
                waypoint_batch_idx = min_dist_idx
                self.closest_node = start + min_dist_idx
            else:
                waypoint_batch_idx = min(min_dist_idx + 1, len(waypoints) - 1)
                self.closest_node = min(start + min_dist_idx + 1, self.goal_node)

            self.sg_node = start + waypoint_batch_idx
            chosen_traj = waypoints[waypoint_batch_idx]

            try:
                sg_img = np.ascontiguousarray(self.topomap[self.sg_node])
                ros_sg_img = self.bridge.cv2_to_imgmsg(sg_img, encoding="rgb8")
                ros_sg_img.header.stamp = self.get_clock().now().to_msg()
                ros_sg_img.header.frame_id = "base_footprint"
                self.subgoal_img_pub.publish(ros_sg_img)
            except Exception as e:
                self.get_logger().error(f"Failed to publish subgoal image: {e}")

            if MODEL_PARAMS["normalize"]:
                chosen_traj[:, :2] *= MAX_V / RATE

            self.chosen_waypoint = chosen_traj[args.waypoint] / (args.waypoint + 1)

            # Waypoint message
            wp_msg = Float32MultiArray()
            wp_msg.data = self.chosen_waypoint.astype(np.float32).tolist()
            self.waypoint_pub.publish(wp_msg)

            try:
                img_np = pil_to_numpy_array(
                    image_input=self.context_queue[-1],
                    target_size=VIZ_IMAGE_SIZE_FISHEYE,
                )
                self._publish_overlay_image(
                    camera_matrix_orig=INTRINSICS,
                    dist_coeffs=DIST_COEFF,
                    img=img_np,
                    pub=self.img_overlay_pub,
                    trajs=chosen_traj[None, :],
                    viz_img_size=VIZ_IMAGE_SIZE_FISHEYE,
                    camera_height=CAMERA_HEIGHT,
                    camera_x_offset=CAMERA_X_OFFSET,
                    resize_factor=False,
                )
            except Exception as e:
                self.get_logger().error(f"Failed to publish overlay image: {e}")

            # Closest node
            self.get_logger().info(f"Closest node: {self.closest_node}")
            cn_msg = Int32()
            cn_msg.data = self.closest_node
            self.closest_node_pub.publish(cn_msg)

            # --- Visualisation: waypoint pose --------------------------------
            stamp = self.get_clock().now().to_msg()

            wp_viz = PoseStamped()
            wp_viz.header.frame_id = "base_link"
            wp_viz.header.stamp = stamp
            wp_viz.pose = Pose(
                position=Point(
                    x=float(self.chosen_waypoint[0]),
                    y=float(self.chosen_waypoint[1]),
                    z=0.0,
                )
            )
            self.waypoint_viz_pub.publish(wp_viz)

            # --- Visualisation: path -----------------------------------------
            path_msg = Path()
            path_msg.header.frame_id = "base_link"
            path_msg.header.stamp = stamp
            for wp in waypoints[waypoint_batch_idx]:
                ps = PoseStamped()
                ps.pose = Pose(position=Point(x=float(wp[0]), y=float(wp[1]), z=0.0))
                path_msg.poses.append(ps)
            self.path_viz_pub.publish(path_msg)

        # ---- Velocity command (always sent) ---------------------------------)

        # ================================

        assert len(self.chosen_waypoint) == 2 or len(self.chosen_waypoint) == 4, (
            "waypoint must be a 2D or 4D vector"
        )
        if len(self.chosen_waypoint) == 2:
            dx, dy = self.chosen_waypoint
        else:
            dx, dy, hx, hy = self.chosen_waypoint
        # this controller only uses the predicted heading if dx and dy near zero
        if len(self.chosen_waypoint) == 4 and np.abs(dx) < EPS and np.abs(dy) < EPS:
            v = 0
            w = self._clip_angle(np.arctan2(hy, hx)) / DT
        elif np.abs(dx) < EPS:
            v = 0
            w = np.sign(dy) * np.pi / (2 * DT)
        else:
            v = dx / DT
            w = np.arctan(dy / dx) / DT
        v = np.clip(v, 0, MAX_V)
        w = np.clip(w, -MAX_W, MAX_W)

        # ================================

        self.vel_msg.linear.x = float(v)
        self.vel_msg.angular.z = float(w)
        self.get_logger().debug(f"Vel cmd: v={v:.3f}  w={w:.3f}")
        self.vel_pub.publish(self.vel_msg)

        # ---- Action history -------------------------------------------------
        # Convert velocity commands back to waypoints for action history
        hist_wp = self.chosen_waypoint.copy()
        hist_wp[0] = v * DT
        hist_wp[1] = (v * DT) * np.tan(w * DT) if abs(v) > 1e-8 else 0.0

        if MODEL_PARAMS["normalize"]:
            hist_wp[0] /= MAX_V / RATE
            hist_wp[1] /= MAX_V / RATE

        self.action_history.append(hist_wp.astype(np.float32))
        if len(self.action_history) > self.context_size:
            self.action_history.pop(0)

        # ---- Goal check -----------------------------------------------------
        self.reached_goal = self.closest_node == self.goal_node
        goal_msg = Bool()
        goal_msg.data = self.reached_goal
        self.goal_pub.publish(goal_msg)

        if self.reached_goal:
            self.get_logger().info("Reached goal! Stopping...")
            self.vel_pub.publish(Twist())  # Stop the robot
            self.timer.cancel()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="GNM DIFFUSION EXPLORATION — ROS2 port"
    )
    parser.add_argument(
        "--model",
        "-m",
        default="VISTA_WITHOUT_AH",
        type=str,
        help="ONNX model name (without .onnx extension)",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=1,
        type=int,
        help="Index of waypoint used for control (default: 1)",
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="bag5_",
        type=str,
        help="Topomap image subdirectory name",
    )
    parser.add_argument(
        "--goal-node",
        "-g",
        default=-1,
        type=int,
        help="Goal node index (-1 = last node)",
    )
    parser.add_argument(
        "--close-threshold",
        "-t",
        default=10,
        type=int,
        help="Distance threshold for advancing the closest node (default: 3)",
    )
    parser.add_argument(
        "--radius",
        "-r",
        default=2,
        type=int,
        help="Topomap look-ahead/behind radius (default: 2)",
    )
    parser.add_argument(
        "--viz-path",
        "-v",
        default="visualizations/model_preds",
        help="Directory for saving visualisation images",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help="Number of sampled actions (default: 8)",
    )

    # rclpy needs to be init'd before we parse args that may include ROS remaps.
    # Use known_args to avoid clashes with ROS2 CLI arguments.
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = None
    try:
        node = NavNode(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[Shutdown] KeyboardInterrupt received...")
    except Exception as e:
        print(f"\n[Shutdown] Unexpected error occurred: {e}")
        if node is not None:
            # 1. Stop all timers/subscriptions/publishers first
            node.destroy_node()
            node.model = None  # Explicitly release model resources before shutdown
            # 2. Delete the node object entirely to drop references
            del node

        # 3. Clear model from memory before rclpy.shutdown
        gc.collect()

        if rclpy.ok():
            rclpy.shutdown()

        os._exit(0)


if __name__ == "__main__":
    main()

import rclpy
from rclpy.node import Node
import numpy as np
from typing import Optional, Tuple
from numpy.typing import NDArray
import os

from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped, Quaternion
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.duration import Duration
import matplotlib.pyplot as plt


from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import PyKDL
from youbot_kinematics.urdf import treeFromUrdfModel
from youbot_kinematics.urdf_parser import URDF

from ament_index_python.packages import get_package_share_directory

import yaml


class TrajDemo(Node):
    def __init__(self):
        super().__init__("traj_demo")

        self.joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self.joint_state_callback, 5
        )
        self.joint_state_sub  # prevent unused variable warning
        self.traj_publisher = self.create_publisher(
            JointTrajectory, "/franka_arm_controller/joint_trajectory", 5
        )
        self.traj_publisher

        # load from urdf file
        self.declare_parameter("urdf_package", "franka_description")
        self.urdf_package = (
            self.get_parameter("urdf_package").get_parameter_value().string_value
        )
        self.urdf_package_path = get_package_share_directory(self.urdf_package)
        self.declare_parameter("urdf_path_in_package", "urdfs/fr3.urdf")
        self.urdf_path_in_package = (
            self.get_parameter("urdf_path_in_package")
            .get_parameter_value()
            .string_value
        )
        self.urdf_name_path = os.path.join(
            self.urdf_package_path, self.urdf_path_in_package
        )

        self.get_logger().info(
            f"loading robot into KDL from urdf: {self.urdf_name_path}"
        )

        robot = URDF.from_xml_file(self.urdf_name_path)

        (ok, self.kine_tree) = treeFromUrdfModel(robot)

        if not ok:
            raise RuntimeError("couldn't load URDF into KDL tree succesfully")

        self.declare_parameter("base_link", "base")
        self.declare_parameter("ee_link", "fr3_link8")
        self.base_link = (
            self.get_parameter("base_link").get_parameter_value().string_value
        )
        self.ee_link = self.get_parameter("ee_link").get_parameter_value().string_value

        self.kine_chain = self.kine_tree.getChain(self.base_link, self.ee_link)
        self.NJoints = self.kine_chain.getNrOfJoints()
        self.current_joint_position = PyKDL.JntArray(self.NJoints)
        self.current_joint_velocity = PyKDL.JntArray(self.NJoints)
        # KDL solvers
        self.ik_solver = PyKDL.ChainIkSolverPos_LMA(self.kine_chain)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # loading trajectory structure

        self.declare_parameter("traj_cfg_pkg", "trajectory_demo")
        self.traj_cfg_pkg = (
            self.get_parameter("traj_cfg_pkg").get_parameter_value().string_value
        )
        self.traj_cfg_pkg_path = get_package_share_directory(self.traj_cfg_pkg)
        self.declare_parameter("traj_cfg_path_within_pkg", "cfg/traj_waypoints.yaml")
        self.traj_cfg_path_within_pkg = (
            self.get_parameter("traj_cfg_path_within_pkg")
            .get_parameter_value()
            .string_value
        )

        self.traj_cfg_path = os.path.join(
            self.traj_cfg_pkg_path, self.traj_cfg_path_within_pkg
        )

        with open(self.traj_cfg_path) as stream:
            self.traj_cfg = yaml.safe_load(stream)

        assert type(self.traj_cfg["joint_names"]) == list, type(
            self.traj_cfg["joint_names"]
        )
        assert len(self.traj_cfg["joint_names"]) == self.NJoints

        self.created_traj = False

        self.get_logger().info(f"got traj cfg:\n{self.traj_cfg}")

        # history of the trajectory, one entry per joint state message
        self.time_hist = []  # seconds since node start
        self.joint_pos_hist = []
        self.joint_vel_hist = []
        self.ee_pos_hist = []

        # cartesian waypoints that were sent, for plotting and for the timer
        self.waypoint_positions = []
        self.traj_start_time = None
        self.traj_duration = None
        self.start_time = self.get_clock().now()

    def joint_state_callback(self, msg: JointState):
        """Callback for the joint states of the robot arm. It will get joint positions and velocities, and eventually the cartesian position of the end-effector and save this. It allows initializes the trajectory once the first joint state message comes through.

        Args:
            msg (JointState): ROS Joint State Message.
        """
        for i in range(len(msg.name)):
            n = msg.name[i]
            # joint state order is not guaranteed to match the KDL chain order
            if n not in self.traj_cfg["joint_names"]:
                continue
            idx = self.traj_cfg["joint_names"].index(n)
            pos = msg.position[i]
            vel = msg.velocity[i]

            self.current_joint_position[idx] = pos
            self.current_joint_velocity[idx] = vel

        joint_pos, joint_vel = self.kdl_to_np(
            self.current_joint_position
        ), self.kdl_to_np(self.current_joint_velocity)

        now = self.get_clock().now()
        self.time_hist.append((now - self.start_time).nanoseconds * 1e-9)
        self.joint_pos_hist.append(joint_pos)
        self.joint_vel_hist.append(joint_vel)

        ee_pos = self.get_ee_pos_ros()
        self.ee_pos_hist.append(ee_pos)

        # we do this after we have our first callback to have current joint positions
        if not self.created_traj:
            joint_traj = self.create_traj()
            self.traj_publisher.publish(joint_traj)
            self.created_traj = True
            self.traj_start_time = now
            return

        # timer: trajectory is done once the end effector is at the last waypoint and has stopped
        if self.traj_duration is None and ee_pos is not None:
            dist_to_goal = np.linalg.norm(
                np.array(ee_pos) - np.array(self.waypoint_positions[-1])
            )
            if dist_to_goal < 5e-3 and np.linalg.norm(joint_vel) < 1e-2:
                self.traj_duration = (now - self.traj_start_time).nanoseconds * 1e-9
                self.get_logger().info(
                    f"trajectory finished in {self.traj_duration:.2f} s "
                    f"(planned {self.goal_times[-1]:.2f} s)"
                )

    def kdl_to_np(self, data: PyKDL.JntArray) -> NDArray:
        """Helper Function to go from KDL arrays to numpy arrays

        Args:
            data (PyKDL.JntArray): desired KDL array to convert

        Returns:
            NDArray: converted NP Array
        """
        is_1d = data.columns() == 1
        np_shape = (data.rows(), data.columns()) if not is_1d else (data.rows(),)
        mat = np.zeros(np_shape)
        for i in range(data.rows()):
            if not is_1d:
                for j in range(data.columns()):
                    mat[i, j] = data[i, j]
            else:
                mat[i] = data[i]
        return mat

    def get_ee_pos_ros(self) -> Tuple[float]:
        """FK function that uses TF2 ROS to do forward kinematics and return world coordinates of the end effector.

        Returns:
            Tuple[float]: position, 3 dimensional. None if the transform is not available yet.
        """
        try:
            t = self.tf_buffer.lookup_transform(
                self.base_link, self.ee_link, rclpy.time.Time()
            )
        except TransformException as ex:
            self.get_logger().debug(
                f"could not transform {self.base_link} to {self.ee_link}: {ex}"
            )
            return None

        pos = [
            t.transform.translation.x,
            t.transform.translation.y,
            t.transform.translation.z,
        ]

        return pos

    def get_joint_pos(
        self,
        current_angles: Tuple[float],
        target_position: Tuple[float],
        target_orientation: Optional[Tuple[float]] = None,
    ) -> Tuple[float]:
        """Helper function to solve inverse kinematics using the KDL library

        Args:
            current_angles (Tuple[float]): current joint angles of the robot arm
            target_position (Tuple[float]): target cartesian position, 3 dimensional, of the end effector in world coordinates
            target_orientation (Optional[Tuple[float]], optional): target orientation in RPY format, optional. Defaults to None.

        Raises:
            RuntimeError: raises error if IK fails to solve. Check the workspace of the robot if this happens.

        Returns:
            Tuple[float]: joint angles that were found to satisfy targets
        """

        assert len(target_position) == 3
        assert target_orientation is None or len(target_orientation) == 3
        assert len(current_angles) == self.NJoints

        pos = PyKDL.Vector(target_position[0], target_position[1], target_position[2])
        if target_orientation is not None:
            # Constructs a rotation by first applying a rotation of r around the x-axis, then a rotation of p around the original y-axis, and finally a rotation of y around the original z-axis
            rot = PyKDL.Rotation.RPY(
                target_orientation[0], target_orientation[1], target_orientation[2]
            )

        seed_array = PyKDL.JntArray(self.NJoints)
        for i in range(self.NJoints):
            seed_array[i] = current_angles[i]

        if target_orientation is not None:
            goal_pose = PyKDL.Frame(rot, pos)
        else:
            goal_pose = PyKDL.Frame(pos)
        result_angles = PyKDL.JntArray(self.NJoints)

        if self.ik_solver.CartToJnt(seed_array, goal_pose, result_angles) >= 0:
            result = list(result_angles)
            return result
        else:
            raise RuntimeError(
                f"Did not solve for goal_pose: {goal_pose} with initial seed {seed_array}"
            )

    def create_traj(self) -> JointTrajectory:
        """Helper function to generate the trajectory to send to the arm.  Loops over cartesian waypoints from the config file, processing them into joint positions using the IK Solver

        Returns:
            JointTrajectory: Ros JointTrajectory to publish
        """
        cartesian_waypoints = self.traj_cfg["waypoints"]["cartesion"]

        cur_joint_pos = self.kdl_to_np(self.current_joint_position)

        goal_positions = []
        goal_times = []

        time_since_start = 0.0

        for waypoint in cartesian_waypoints:
            pos = waypoint["position"]
            time_sec = waypoint["time"] + time_since_start
            assert type(time_sec) == float, type(time_sec)
            assert len(pos) == 3
            goal_times.append(time_sec)
            # time_sec is already cumulative
            time_since_start = time_sec

            cur_joint_pos = self.get_joint_pos(cur_joint_pos, pos)
            goal_positions.append(cur_joint_pos)
            self.waypoint_positions.append(pos)

        self.goal_times = goal_times

        joint_traj = JointTrajectory()
        joint_traj.header.stamp = self.get_clock().now().to_msg()
        joint_traj.header.frame_id = self.base_link
        joint_traj.joint_names = self.traj_cfg["joint_names"]

        for joint_pos, t in zip(goal_positions, goal_times):
            point = JointTrajectoryPoint()
            point.positions = [float(q) for q in joint_pos]
            point.time_from_start = Duration(seconds=t).to_msg()
            joint_traj.points.append(point)

        self.get_logger().info(
            f"sending trajectory with {len(joint_traj.points)} points, planned duration {goal_times[-1]:.2f} s"
        )
        return joint_traj

    def plot_joint_traj(self):
        """Helper function to plot the desired results from following the trajectory. Will be called once the process is interupted via a Keyboard Interupt."""

        ee_pos = np.array([p for p in self.ee_pos_hist if p is not None])
        if len(ee_pos) == 0:
            self.get_logger().warn("no end effector positions recorded, nothing to plot")
            return
        waypoints = np.array(self.waypoint_positions)

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.plot(ee_pos[:, 0], ee_pos[:, 1], ee_pos[:, 2], label="end effector (TF2)")
        ax.scatter(ee_pos[0, 0], ee_pos[0, 1],ee_pos[0, 2], c="g", label="start")
        ax.scatter(
            waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], c="r", s=8, label="waypoints"
        )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        title = os.path.basename(self.traj_cfg_path)
        if self.traj_duration is not None:
            title += f" - {self.traj_duration:.2f} s"
        ax.set_title(title)
        ax.legend()

        fig_path = os.path.splitext(os.path.basename(self.traj_cfg_path))[0] + "_plot.png"
        fig.savefig(fig_path, dpi=200)
        self.get_logger().info(f"saved plot to {os.path.abspath(fig_path)}")

        plt.show()


def main(args=None):
    rclpy.init(args=args)
    traj_demo = TrajDemo()

    try:
        rclpy.spin(traj_demo)
    except KeyboardInterrupt:
        pass

    traj_demo.plot_joint_traj()
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    traj_demo.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":

    main()

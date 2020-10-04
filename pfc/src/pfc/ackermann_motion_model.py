import rospy

import numpy as np
from numpy import sin, cos, tan, sqrt, arctan2

from ackermann_msgs.msg import AckermannDrive
from geometry_msgs.msg import Pose, PoseArray, Quaternion

class AckermannMotionModel:
    """
    Class for forward-simulating ackermann-steered vehicles. Using it to generate paths to follow.
    """
    def __init__(self, veh_length=0.29, veh_width=0.23):
        """
        Args:
            veh_length: The distance between the rear and front axles
            veh_width: The length of the front axle
        """
        self.veh_width = veh_width
        self.veh_length = veh_length

    def sample_ackermann_path(self, from_pose, n_steps=100, resample_every=10, v_lim=(-1.0, 1.0), max_steer=0.3, x_lim = (-2., 2.), y_lim = (-2., 2.), dt=0.1):
        """
        For testing pfc. Get a list of waypoints from randomly sampled Ackermann actions.
        Args:
            from_pose: some Pose()
            n_steps: Number of timesteps to sample
            resample_every = use the same command for this many timesteps
            v_lim: tuple of (minimum velocity, maximum_velocity)
            max_steer: max acceptable steering angle (assumed symmetric)
        Returns:
            list of Pose()
        """
        def valid_pose(pose):
            return pose.position.x > x_lim[0] and pose.position.x < x_lim[1] and pose.position.y > y_lim[0] and pose.position.y < y_lim[1]

        if not valid_pose(from_pose):
            print('Sampled path from invalid region')
            return PoseArray(poses=[from_pose])

        poses = [from_pose]
        print('FROM POSE')
        print(from_pose)
        cnt = 0
        while cnt < n_steps/resample_every:
            cmd = self.sample_action(v_lim=v_lim, max_steer=max_steer)
            check_poses = [poses[-1]]
            for t in range(resample_every):
                check_poses.append(self.forward(check_poses[-1], cmd, dt))
            if all([valid_pose(p) for p in check_poses]):
                cnt += 1
                poses.extend(check_poses)

        return PoseArray(poses = poses)

    def sample_action(self, v_lim=(-1.0, 1.0), max_steer=0.3):
        vel = np.random.uniform(v_lim[0], v_lim[1])
        steer = np.random.uniform(-max_steer, max_steer)
        return AckermannDrive(speed=vel, steering_angle=steer)

    def forward(self, pose, cmd, dt):
        """
        Forward simulates from pose the cmd for dt seconds. Returns a new pose
        """
        x = pose.position.x
        y = pose.position.y
        theta = self.quat_2_yaw(pose.orientation)

        vel = cmd.speed
        steer = cmd.steering_angle
        dx = vel * cos(theta)
        dy = vel * sin(theta)
        dtheta = (vel/self.veh_length) * tan(steer)

        x_new = dt * dx + x
        y_new = dt * dy + y
        th_new = dt * dtheta + theta
        out = Pose()
        out.position.x = x_new
        out.position.y = y_new
        out.orientation = self.yaw_2_quat(th_new)

        return out
        
    def quat_2_yaw(self, orientation):
        qw = orientation.w
        qx = orientation.x
        qy = orientation.y
        qz = orientation.z
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2 * (qy*qy + qz*qz))
        return yaw

    def yaw_2_quat(self, yaw):
        """
        Assumed that roll, pitch = 0
        """
        quat = Quaternion()
        quat.w = cos(yaw/2)
        quat.x = 0.
        quat.y = 0.
        quat.z = sin(yaw/2)
        return quat
        

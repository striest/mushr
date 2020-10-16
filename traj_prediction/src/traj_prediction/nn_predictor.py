import rospy
import numpy as np
import torch
import os
import skimage
import matplotlib.pyplot as plt

from numpy import sin, cos, pi, sqrt
from skimage.transform import ProjectiveTransform

from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, Quaternion

class NNPredictor:
    """
    Take the trained neural net and use it to predict next steps.
    This module needs a stream of:
        1. Heightmap
        2. Current state
        3. Plan
    and will produce 10 steps of predicted traj.
    """
    def __init__(self):
        self.heightmap = None
        self.heightmap_metadata = None
        self.curr_state = np.zeros(4)
        self.plan = None
        self.prediction = None

        self.network = torch.load('src/traj_prediction/networks/network_itr400.cpt', map_location='cpu')
        print(self.network)

    def handle_heightmap(self, msg):
        shape = (msg.info.width, msg.info.height)
        arr = np.array(msg.data).astype(float) / 100. #Assuming heightmap is in cm.
        self.heightmap = np.reshape(arr, shape)
        self.heightmap_metadata = msg.info
        #We have to adjust the nn's scaling to the right ranges
        #Network normalizes position, angle, velocity. We need to correct the position scaling.
        #Assume that mean position is the mean of the heightmap value
        #Assume std from uniform over the range.
        x_min = self.heightmap_metadata.origin.position.x
        y_min = self.heightmap_metadata.origin.position.y
        res = self.heightmap_metadata.resolution
        x_max = self.heightmap_metadata.width * res + x_min
        y_max = self.heightmap_metadata.height * res + y_min

        self.network.data_shift[0, 0, 0] = (x_min + x_max) / 2
        self.network.data_shift[0, 0, 1] = (y_min + y_max) / 2
        self.network.data_scale[0, 0, 0] = (x_max - x_min) / sqrt(12)
        self.network.data_scale[0, 0, 1] = (y_max - y_min) / sqrt(12)

    def handle_pose(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        th = self.quat_2_yaw(msg.pose)
        self.curr_state[:3] = np.array([x, y, th])
        
    def handle_odom(self, msg):
        #Particle filter doesnt give me velocity.
        vx = msg.twist.twist.linear.x
        self.curr_state[3] = vx

    def handle_plan(self, msg):
        """
        Plan is a PoseArray. We want a numpy array
        """
        out = []
        for p in msg.poses:
            x = p.position.x
            y = p.position.y
            th = self.quat_2_yaw(p)
            out.append(np.array([x, y, th]))
        self.plan = np.stack(out, axis=0)

    def get_cropped_heightmap(self):
        """
        Gets the correct crop of the heightmap
        """
        map_x, map_y = self.pose_2_map(self.curr_state[:3])
        th = self.curr_state[2]
        windowsize = 100

        #OK, to properly get the heightmap, we need to 1. translate to vehicle origin. 2. Rotate by theta 3. translate by -windowsize/2 to center
        HTM_trans = np.array([[1., 0., map_x], [0., 1., map_y], [0., 0., 1.]])
        HTM_rot = np.array([[cos(th), -sin(th), 0.], [sin(th), cos(th), 0.], [0., 0., 1.]])
        HTM_center = np.array([[1., 0., -windowsize//2], [0., 1., -windowsize//2], [0., 0., 1.]])
        HTM = np.matmul(HTM_trans, np.matmul(HTM_rot, HTM_center))
        heightmap_tr = skimage.transform.warp(self.heightmap, ProjectiveTransform(matrix=HTM))
        heightmap_out = heightmap_tr[:windowsize, :windowsize]

        return heightmap_out 

    def get_plan_segment(self):
        """
        Find the closest path point, and take that and the next nine.
        """
        pos = np.expand_dims(self.curr_state[:2], axis=0)
        path_locs = self.plan[:, :2]
        dists = np.linalg.norm(path_locs - pos)
        pt_idx = np.argmin(dists)
        if len(self.plan) - pt_idx > 10:
            segment = self.plan[pt_idx:pt_idx+10, :]
        else:
            segment = self.plan[pt_idx:, :]
            nrepeats = 10 - (len(self.plan) - pt_idx)
            pad = np.repeat(np.expand_dims(self.plan[-1], axis=0), repeats = nrepeats, axis=0)
            segment = np.concatenate([segment, pad], axis=0)

        return segment

    @property
    def can_predict(self):
        return self.heightmap is not None and self.plan is not None and self.curr_state is not None

    def predict(self):
        heightmap_np = self.get_cropped_heightmap()
        plan_np = self.get_plan_segment()
        state_np = np.copy(self.curr_state)

        heightmap_torch = torch.tensor(skimage.transform.resize(heightmap_np, [24, 24])).unsqueeze(0).unsqueeze(0).float()
        plan_torch = torch.tensor(plan_np).unsqueeze(0).float()
        state_torch = torch.tensor(state_np).unsqueeze(0).unsqueeze(0).float()

        with torch.no_grad():
            preds_torch = self.network.forward({'state':state_torch, 'heightmap':heightmap_torch, 'traj_plan':plan_torch})

        print(preds_torch)

        self.prediction = preds_torch.squeeze().numpy()

    @property
    def predict_msg(self):
        if self.prediction is None:
            return PoseArray() 

        poses = []
        for state in self.prediction:
            p = Pose()
            p.position.x = state[0]
            p.position.y = state[1]
            p.orientation = self.yaw_2_quat(state[2])
            poses.append(p)

        msg = PoseArray()
        msg.poses = poses
        msg.header.frame_id='map'
        return msg

    def pose_2_map(self, pose):
        """
        Finds the cell indexes of the occupancy grid closest to the pose.
        """
        o_map_x = self.heightmap_metadata.origin.position.x
        o_map_y = self.heightmap_metadata.origin.position.y
        o_map_yaw = self.quat_2_yaw(self.heightmap_metadata.origin)

        assert o_map_yaw == 0., 'TODO: Use the coordinate tranform from the map frame'
        #px = self.pose.position.x * cos(o_map_yaw) - self.pose.position.y * -sin(o_map_yaw)
        #py = self.pose.position.x * sin(o_map_yaw) - self.pose.position.y * cos(o_map_yaw)

        px = pose[0] #note: pose is [x, y, theta], not a pose message.
        py = pose[1]
        
        px = px - o_map_x
        py = py - o_map_y

        px = px//self.heightmap_metadata.resolution
        py = py//self.heightmap_metadata.resolution

        return np.array([px, py])

    def quat_2_yaw(self, pose):
        #Gets yaw from pose
        qw = pose.orientation.w
        qx = pose.orientation.x
        qy = pose.orientation.y
        qz = pose.orientation.z
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2 * (qy*qy + qz*qz))
        return yaw

    def yaw_2_quat(self, yaw):
        q = Quaternion()
        q.w = cos(yaw/2)
        q.x = 0.
        q.y = 0.
        q.z = sin(yaw/2)
        return q

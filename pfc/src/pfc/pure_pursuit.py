import rospy

import numpy as np
from numpy import sin, cos, arctan2, sqrt, pi

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Pose, PoseStamped, PoseArray

class PurePursuitController:
    """
    Implements pure-pursuit for an Ackermann-steered vehicle.
    """
    def __init__(self, lookahead=1.0, max_v=0.5, kp=0.2):
        """
        Args:
            lookahead: The location on the traj to drive to
            velocity: The max speed for pfc
        """
        self.lookahead_dist = lookahead
        self.max_v = max_v
        self.path = np.array([])
        self.path_point = np.zeros(2)
        self.path_point_idx = -1
        self.lookahead_point = np.zeros(2)
        self.kp = kp
        self.pose = Pose()

    def handle_pose(self, msg):
        self.pose = msg.pose

    def handle_path(self, msg):
        acc = []
        for pose in msg.poses:
            x = pose.position.x
            y = pose.position.y
            acc.append(np.array([x, y]))

        self.path = np.stack(acc, axis=0)

    def get_action(self):
        ego_x = self.pose.position.x
        ego_y = self.pose.position.y
        ego_yaw = self.quat_2_yaw(self.pose) #This val is yaw from x_origin to ego_y_ax

        self.path_point, self.path_point_idx = self.get_path_point()
        self.lookahead_point = self.get_lookahead_point()

        #Dist to lookahead
        l_disp = self.lookahead_point - np.array([ego_x, ego_y])
        print('Lookahead disp =\n{}'.format(l_disp))
        l_dist = np.hypot(l_disp[0], l_disp[1])
        print('Lookahead_dist =\n{}'.format(l_dist))
        l_ang = np.arctan2(l_disp[1], l_disp[0]) - ego_yaw
        l_ang = l_ang % (2*pi)
        print('Lookahead_ang (deg) =\n{}'.format(l_ang * (180/pi)))

        x_disp = l_dist * cos(l_ang)
        y_disp = l_dist * sin(l_ang)
        print('X-disp to lookahead =\n{}'.format(x_disp))
        print('Y-disp to lookahead =\n{}'.format(y_disp))
        u_angle = (2 * y_disp) / (l_dist ** 2)
        u_velocity = self.kp * (l_dist / self.lookahead_dist) * self.max_v
        if u_velocity < 0.3: #An inelegant way of dealing w/ the VESC deadzone.
            u_velocity = 0.3 if l_dist > 0.05 else 0.

        print('Commanded V =\n{}'.format(u_velocity))
        print('Commanded ang =\n{}'.format(u_angle))

        u_msg = AckermannDriveStamped()
        u_msg.drive.speed = u_velocity
        u_msg.drive.steering_angle = u_angle

        return u_msg
        

    def get_path_point(self):
        """
        Finds the path point closest to the robot
        """
        if len(self.path) == 0:
            return np.zeros(2), -1
        ego_x = self.pose.position.x
        ego_y = self.pose.position.y
        ego_pose = np.array([[ego_x, ego_y]])
        disps = (ego_pose - self.path)
        dists = np.hypot(disps[:, 0], disps[:, 1])
        path_point_idx = np.argmin(dists)
        path_point = self.path[path_point_idx]
        return path_point, path_point_idx

    def get_lookahead_point(self):
        """
        Moves forward 1 lookahead distance on the path from the current path point
        """
        if self.path_point_idx == len(self.path) - 1 or self.path_point_idx == -1:
            #End of path, no more lookahead
            return self.path_point

        prev_pt = self.path[self.path_point_idx]
        curr_pt = self.path[self.path_point_idx + 1]
        pt_dist = np.hypot((prev_pt - curr_pt)[0], (prev_pt - curr_pt)[1])
        curr_dist = pt_dist
        c = self.path_point_idx
        while curr_dist < self.lookahead_dist and c < len(self.path) - 1:
            prev_pt = self.path[c]
            curr_pt = self.path[c + 1]
            pt_dist = np.hypot((prev_pt - curr_pt)[0], (prev_pt - curr_pt)[1])
            curr_dist += pt_dist
            c += 1

        if curr_dist < self.lookahead_dist:
            return self.path[-1]
        else:
            #Interpolate to get the actual lookahead point
            frac = (curr_dist - self.lookahead_dist) / pt_dist
            pt = frac * prev_pt + (1-frac) * curr_pt
            return pt
            

    def path_point_msg(self):
        msg = Pose()
        msg.position.x = self.path_point[0]
        msg.position.y = self.path_point[1]
        return msg

    def lookahead_point_msg(self):
        msg = Pose()
        msg.position.x = self.lookahead_point[0]
        msg.position.y = self.lookahead_point[1]
        return msg

    def quat_2_yaw(self, pose):
        #Gets yaw from pose
        qw = pose.orientation.w
        qx = pose.orientation.x
        qy = pose.orientation.y
        qz = pose.orientation.z
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2 * (qy*qy + qz*qz))
        return yaw

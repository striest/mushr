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
        self.curr_v = 0.
        self.path = np.array([])
        self.path_point = np.ones(2) * float('inf')
        self.path_point_idx = -1
        self.lookahead_point = np.ones(2) * float('inf')
        self.kp = kp
        self.pose = Pose()
        self.paths = []
        self.current_path = np.array([])
        self.new_path = True
        self.path_idx = 0

    def handle_exec(self, msg):
        self.new_path = msg.data

    def handle_odom(self, msg):
        self.curr_v = msg.twist.twist.linear.x

    def handle_pose(self, msg):
        self.pose = msg.pose
        if np.any(np.isinf(self.path_point)):
            self.path_point = np.array([self.pose.position.x, self.pose.position.y])
        if np.any(np.isinf(self.lookahead_point)):
            self.lookahead_point = np.array([self.pose.position.x, self.pose.position.y])

    def handle_path(self, msg):
        acc = []
        for pose in msg.poses:
            x = pose.position.x
            y = pose.position.y
            acc.append(np.array([x, y]))

        if acc:
            self.path = np.stack(acc, axis=0)
            self.paths = self.split_path(self.path)
            if self.new_path:
                self.path_idx = 0
                self.current_path = self.paths[self.path_idx]
        self.path_point_idx = 0
        #print(self.path)
        self.new_path = False

    def split_path(self, path):
        """
        Split paths into forward and reverse segments. Look for angles of pi/2 or more (positive dot product).
        """
#        import pdb;pdb.set_trace()
        if len(path) < 2:
            return [path]

        paths = [[path[0]]]
        for i in range(1, len(path)-1):
            p0 = path[i-1]
            p1 = path[i]
            p2 = path[i+1]
            v1 = p0 - p1
            v2 = p2 - p1
            d = np.sum(v1 * v2)
            if d > 0:
                print('found cusp')
                paths.append([])
            paths[-1].append(p1)
        paths[-1].append(path[-1])
        paths = [np.stack(p, axis=0) for p in paths] 
        return paths

    def get_action(self):
        ego_x = self.pose.position.x
        ego_y = self.pose.position.y
        ego_yaw = self.quat_2_yaw(self.pose) #This val is yaw from x_origin to ego_y_ax

        self.path_point, self.path_point_idx = self.get_path_point()
#        print(self.path_point, self.path_point_idx, len(self.current_path), len(self.paths))
        self.lookahead_point = self.get_lookahead_point()

        #Dist to lookahead
        l_disp = self.lookahead_point - np.array([ego_x, ego_y])
#        print('Lookahead disp =\n{}'.format(l_disp))
        l_dist = np.hypot(l_disp[0], l_disp[1])
#        print('Lookahead_dist =\n{}'.format(l_dist))
        l_ang = np.arctan2(l_disp[1], l_disp[0]) - ego_yaw
        l_ang = l_ang % (2*pi)
#        print('Lookahead_ang (deg) =\n{}'.format(l_ang * (180/pi)))

        x_disp = l_dist * cos(l_ang)
        y_disp = l_dist * sin(l_ang)
#        print('X-disp to lookahead =\n{}'.format(x_disp))
#        print('Y-disp to lookahead =\n{}'.format(y_disp))
        u_angle = (2 * y_disp) / (l_dist ** 2)
        u_velocity = self.kp * (l_dist / self.lookahead_dist) * self.max_v

        if u_velocity < 0.3: #An inelegant way of dealing w/ the VESC deadzone.
            u_velocity = 0.3 if abs(x_disp) > 0.05 else 0.

        if l_ang > (pi/2) and l_ang < 3*pi/2:
            u_velocity *= -1

        if l_dist < 0.1:
            u_angle = u_velocity = 0.

            if self.path_idx < len(self.paths)- 1: #
                print('changing paths')
                self.path_idx += 1
                self.current_path = self.paths[self.path_idx]
                self.path_point_idx = 0

#        print('Commanded V =\n{}'.format(u_velocity))
#        print('Commanded ang =\n{}'.format(u_angle))

        u_msg = AckermannDriveStamped()
        u_msg.drive.speed = u_velocity
        u_msg.drive.steering_angle = u_angle

        return u_msg
        
    def get_path_point(self):
        """
        Finds the path point closest to the robot
        """
        if len(self.current_path) == 0:
            return np.zeros(2), -1
        ego_x = self.pose.position.x
        ego_y = self.pose.position.y
        ego_pose = np.array([[ego_x, ego_y]])
        disps = (ego_pose - self.current_path)
        dists = np.hypot(disps[:, 0], disps[:, 1])
        path_point_idx = np.argmin(dists[self.path_point_idx:]) + self.path_point_idx
        path_point = self.current_path[path_point_idx]
        return path_point, path_point_idx


    def get_lookahead_point(self):
        """
        Moves forward 1 lookahead distance on the path from the current path point
        """
        lookahead_target_dist = self.lookahead_dist #+ (1 + self.curr_v)

        if self.path_point_idx == len(self.current_path) - 1 or self.path_point_idx == -1:
            #End of path, no more lookahead
            return self.path_point

        prev_pt = self.current_path[self.path_point_idx]
        curr_pt = self.current_path[self.path_point_idx + 1]
        pt_dist = np.hypot((prev_pt - curr_pt)[0], (prev_pt - curr_pt)[1])
        curr_dist = pt_dist
        c = self.path_point_idx
        while curr_dist < lookahead_target_dist and c < len(self.current_path) - 1:
            prev_pt = self.current_path[c]
            curr_pt = self.current_path[c + 1]
            pt_dist = np.hypot((prev_pt - curr_pt)[0], (prev_pt - curr_pt)[1])
            curr_dist += pt_dist
            c += 1

        if curr_dist < lookahead_target_dist:
            return self.current_path[-1]
        else:
            #Interpolate to get the actual lookahead point
            frac = (curr_dist - lookahead_target_dist) / pt_dist
            pt = frac * prev_pt + (1-frac) * curr_pt
            return pt
            

    def path_point_msg(self):
        msg = Pose()
        msg.position.x = self.path_point[0]
        msg.position.y = self.path_point[1]
        out = PoseStamped(pose=msg)
        out.header.frame_id = "/map"
        return out

    def lookahead_point_msg(self):
        msg = Pose()
        msg.position.x = self.lookahead_point[0]
        msg.position.y = self.lookahead_point[1]
        out = PoseStamped(pose=msg)
        out.header.frame_id = "/map"
        return out

    def quat_2_yaw(self, pose):
        #Gets yaw from pose
        qw = pose.orientation.w
        qx = pose.orientation.x
        qy = pose.orientation.y
        qz = pose.orientation.z
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2 * (qy*qy + qz*qz))
        return yaw

class PurePursuitFixedVelocityController(PurePursuitController):
    """
    Pure pursuit, but now we have it traverse the path at a fixed speed.
    Heuristically, start to slow down if you're within 1/2 lookahead distance
    """
    def __init__(self, lookahead=1.0, v=0.6, kp=0.2):
        self.lookahead_scale = lookahead
        self.lookahead_dist = self.lookahead_scale * v
        self.max_v = v
        self.curr_v = 0.
        self.path = np.array([])
        self.path_point = np.ones(2) * float('inf')
        self.path_point_idx = -1
        self.lookahead_point = np.ones(2) * float('inf')
        self.kp = kp
        self.pose = Pose()
        self.paths = []
        self.current_path = np.array([])
        self.new_path = True

    def handle_vel(self, msg):
        self.max_v = msg.data
        self.lookahead_dist = self.lookahead_scale * self.max_v

    def get_action(self):
        ego_x = self.pose.position.x
        ego_y = self.pose.position.y
        ego_yaw = self.quat_2_yaw(self.pose) #This val is yaw from x_origin to ego_y_ax

        self.path_point, self.path_point_idx = self.get_path_point()
#        print(self.path_point, self.path_point_idx, len(self.current_path), len(self.paths))
        self.lookahead_point = self.get_lookahead_point()

        #Dist to lookahead
        l_disp = self.lookahead_point - np.array([ego_x, ego_y])
#        print('Lookahead disp =\n{}'.format(l_disp))
        l_dist = np.hypot(l_disp[0], l_disp[1])
#        print('Lookahead_dist =\n{}'.format(l_dist))
        l_ang = np.arctan2(l_disp[1], l_disp[0]) - ego_yaw
        l_ang = l_ang % (2*pi)
#        print('Lookahead_ang (deg) =\n{}'.format(l_ang * (180/pi)))

        x_disp = l_dist * cos(l_ang)
        y_disp = l_dist * sin(l_ang)
#        print('X-disp to lookahead =\n{}'.format(x_disp))
#        print('Y-disp to lookahead =\n{}'.format(y_disp))
        u_angle = (2 * y_disp) / (l_dist ** 2)
        if l_dist > self.lookahead_dist/2:
            u_velocity = self.max_v
        else:
            u_velocity = self.kp * (l_dist / self.lookahead_dist) * self.max_v

        if u_velocity < 0.3: #An inelegant way of dealing w/ the VESC deadzone.
            u_velocity = 0.3 if abs(x_disp) > 0.05 else 0.

        if l_ang > (pi/2) and l_ang < 3*pi/2:
            u_velocity *= -1

        if l_dist < 0.1:
            u_angle = u_velocity = 0.

            if len(self.paths) > 1: #
                print('changing paths')
                self.paths.pop(0)
                self.current_path = self.paths[0]
                self.path_point_idx = 0

#        print('Commanded V =\n{}'.format(u_velocity))
#        print('Commanded ang =\n{}'.format(u_angle))

        u_msg = AckermannDriveStamped()
        u_msg.drive.speed = u_velocity
        u_msg.drive.steering_angle = u_angle

        return u_msg

import rospy
import numpy as np
import matplotlib.pyplot as plt

from numpy import sin, cos, pi

class DataCollector:
    """
    This class reads in a bunch of ros messages from various topics and makes them into a dataset
    Subscribed topics:
        static (read once):
            /heightmap: The heightmap for this traj
            /car/planner/plan: The plan for this trial
        dynamic (updated every cycle) (both of these are timestamped, so we can do interpolation as necessary)
            /car/vesc/odom: Get the velocity at the timestep 
            /car/particle_filter/inferred_pose: Get the pose
    """
    def __init__(self, frequency):
        self.frequency = frequency
        self.heightmap = None
        self.heightmap_metadata = None
        self.plan = None
        self.velocities = [] #Store linear velocity, timestamp
        self.poses = [] #Store x, y, theta and tiemstamp

    @property
    def can_build(self):
        return not (self.heightmap is None or self.plan is None or len(self.velocities) == 0 or len(self.poses) == 0)

    def build_training_sample(self):
        """
        Given current stored data, do interpolation to build a training sample at the frequency given.
        Algo:
            Find the channel that starts later - pose or odom. Set that start time to be 0.
            Do the same for the end time.
            Given those bounds, interpolate your values into the correct number of datapoints.
        """
        poses = np.array(self.poses)
        vels = np.array(self.velocities)
        start_time = max(poses[0, 3], vels[0, 1])
        stop_time = min(poses[-1, 3], vels[-1, 1])
        poses[:, 3] -= start_time
        vels[:, 1] -= start_time

        #We need to take our samples at these timesteps. Let's linearly interpolate between surrounding points.
        #I'm also just going to assume that the first step is aligned
        t_samples = np.arange(0, stop_time - start_time, 1./self.frequency)

        pose_interp = np.stack([np.interp(t_samples, xp=poses[:, 3], fp = poses[:, i]) for i in range(3)], axis=1)
        vel_interp =  np.stack([np.interp(t_samples, xp=vels[:, 1], fp = vels[:, i]) for i in range(1)], axis=1)

        pose_out = self.transform_traj(pose_interp)
        pose_out = self.smooth_traj(pose_out, window=3)
        plan_out = self.transform_traj(self.plan)
        vel_out = vel_interp * 5.0

        #Match the format of the preprocessed training sample. i.e. get f{x, y, yaw}, r{x, y, yaw}, fintersect, fspeed, heightmap.
        plt.imshow(self.heightmap, origin='lower')
        plt.plot(plan_out[:, 0], plan_out[:, 1], c='b', label='Plan')
        plt.plot(pose_out[:, 0], pose_out[:, 1], c='r', label='GT')
        plt.scatter(pose_out[0, 0], pose_out[0, 1], c='r', marker='x', label='Start')
        plt.legend()
        plt.show()

        out_dict = {
            'rx':plan_out[:, 0],
            'ry':plan_out[:, 1],
            'ryaw':plan_out[:, 2],
            'fx':pose_out[:, 0],
            'fy':pose_out[:, 1],
            'fyaw':pose_out[:, 2],
            'fspeed':vel_out[:, 0],
            'fintersect':np.arange(len(pose_out)),
            'heightmap':self.heightmap
        }
        return out_dict

    def smooth_traj(self, traj, window=1):
        """
        The localization is a bit noisy. Let's smooth it.
        Take sliding average over all channels
        """
        kernel = np.ones(2*window + 1)
        kernel /= kernel.size
        traj_start = np.copy(traj[0])
        traj_end = np.copy(traj[-1])
        to_conv = np.concatenate([np.array([traj_start]*window), traj, np.array([traj_end]*window)], axis=0)
        traj_conv = np.stack([np.convolve(to_conv[:, i], kernel, mode='valid') for i in range(traj.shape[1])], axis=1)
        traj_conv[0] = traj_start
        traj_conv[-1] = traj_end
        return traj_conv

    def transform_traj(self, traj):
        """
        Use the metadata from the heightmap to transform trajs into the frame of the map.
        Also rescale units from meters to heightmap units
        Assuming traj is passed in as an N x 3 tensor of [timedim x [x, y, th]]
        """
        map_x = self.heightmap_metadata.origin.position.x
        map_y = self.heightmap_metadata.origin.position.y
        map_th = self.quat_2_yaw(self.heightmap_metadata.origin)
        map_r = self.heightmap_metadata.resolution

        rot_HTM = np.array([[cos(-map_th), -sin(-map_th), 0.], [sin(-map_th), cos(-map_th), 0.], [0., 0., 1.]])
        tr_HTM = np.array([[1., 0., -map_x], [0., 1., -map_y], [0., 0., 1.]])
        HTM = np.matmul(rot_HTM, tr_HTM)

        traj_homogeneous = np.concatenate([traj[:, :2], np.ones([len(traj), 1])], axis=1)
        traj_tr = np.matmul(HTM, traj_homogeneous.T).T
        traj_tr[:, 2] = traj[:, 2] - map_th
        traj_tr[:, :2] /= map_r
        return traj_tr

    def handle_heightmap(self, msg):
        if self.heightmap is None or self.heightmap_metadata is None:
            shape = (msg.info.width, msg.info.height)
            arr = np.array(msg.data).astype(float) / 100. #Assuming heightmap is in cm.
            self.heightmap = np.reshape(arr, shape)
            self.heightmap_metadata = msg.info

    def handle_plan(self, msg):
        """
        Plan is a PoseArray. We want a numpy array
        """
        if (self.plan is None or len(self.plan) == 0) and len(msg.poses) > 0:
            out = []
            for p in msg.poses:
                x = p.position.x
                y = p.position.y
                th = self.quat_2_yaw(p)
                out.append(np.array([x, y, th]))
            self.plan = np.stack(out, axis=0)

    def handle_odom(self, msg):
        t = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9
        vx = msg.twist.twist.linear.x
        self.velocities.append([vx, t])

    def handle_pose(self, msg):
        t = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9
        x = msg.pose.position.x
        y = msg.pose.position.y
        th = self.quat_2_yaw(msg.pose)
        self.poses.append([x, y, th, t])

    def quat_2_yaw(self, pose):
        #Gets yaw from pose
        qw = pose.orientation.w
        qx = pose.orientation.x
        qy = pose.orientation.y
        qz = pose.orientation.z
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2 * (qy*qy + qz*qz))
        return yaw

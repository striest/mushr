#!/usr/bin/env python

import rospy
import numpy as np

from data_collection.data_collector import DataCollector

from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from nav_msgs.msg import Odometry, OccupancyGrid

def main():
    rospy.init_node("data_collection_node")

    collector = DataCollector(frequency = 20)

    heightmap_sub = rospy.Subscriber('/heightmap', OccupancyGrid, collector.handle_heightmap)
    plan_sub = rospy.Subscriber('/car/planner/path', PoseArray, collector.handle_plan)
    pose_sub = rospy.Subscriber('/car/particle_filter/inferred_pose', PoseStamped, collector.handle_pose)
    odom_sub = rospy.Subscriber('/car/vesc/odom', Odometry, collector.handle_odom)

    path = raw_input('Input the location to save (including filename). Drive the car/play the rosbag, then press enter to build the training example once it finishes:\n')

    data = collector.build_training_sample()

    if path != '':
        np.savez(path, **data)

if __name__ == '__main__':
    main()

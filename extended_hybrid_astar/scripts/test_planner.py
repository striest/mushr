#!/usr/bin/env python

import rospy

from geometry_msgs.msg import PoseArray, Pose

import numpy as np
import matplotlib.pyplot as plt
from extended_hybrid_astar.astarVREP import *


if __name__ == '__main__':
    rospy.init_node('ehas_planner')
    rate = rospy.Rate(10)
    path_pub = rospy.Publisher('car/planner/path', PoseArray, queue_size=1)

    data = np.load('0_0_0_0.npz')
    heightmap = data['heightmap']
    angle_map = heightmap.copy()

    output = plan_from_VREP(heightmap, 100, 360, -3.14, 60, 190, heightmap)

    poses = []
    for x, y in zip(output[0], output[1]):
        p = Pose()
        p.position.x = x
        p.position.y = y
        poses.append(p)

    msg = PoseArray(poses = poses)

    while not rospy.is_shutdown():
        path_pub.publish(msg)
        rate.sleep()

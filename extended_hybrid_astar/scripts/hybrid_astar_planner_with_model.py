#!/usr/bin/env python

import rospy
import torch

from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from nav_msgs.msg import MapMetaData, OccupancyGrid
from std_msgs.msg import Bool, Float32

import numpy as np
import matplotlib.pyplot as plt

from extended_hybrid_astar.hybrid_astar_planner_with_model import HybridAStarPlannerWithModel

if __name__ == '__main__':
    rospy.init_node('hybrid_astar_planner_with_model')
    net = torch.load('src/extended_hybrid_astar/src/extended_hybrid_astar/hybrid_astar/network_itr200.cpt', map_location=torch.device('cpu'))
    net = torch.load('networks/network_itr1.cpt', map_location=torch.device('cpu'))
    net.eval()
    print(net)

    planner = HybridAStarPlannerWithModel(net)

    rate = rospy.Rate(10)
    path_pub = rospy.Publisher('/car/planner/path', PoseArray, queue_size=1)
    vel_pub = rospy.Publisher('/car/planner/desired_velocity', Float32, queue_size=1)

    pose_sub = rospy.Subscriber('/car/particle_filter/inferred_pose', PoseStamped, planner.handle_pose)
    #pose_sub = rospy.Subscriber('/car/car_pose', PoseStamped, planner.handle_pose)
    #goal_sub = rospy.Subscriber('/car/planner/goal', PoseStamped, planner.handle_goal)

    goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, planner.handle_goal)
    map_sub = rospy.Subscriber('/map', OccupancyGrid, planner.handle_occupancy_grid)
    heightmap_sub = rospy.Subscriber('/heightmap', OccupancyGrid, planner.handle_heightmap)
    exec_sub = rospy.Subscriber('/car/executive/reached_goal', Bool, planner.handle_reached_goal)

    rate.sleep()

    while not rospy.is_shutdown():
        print(planner.should_plan)

        if planner.should_plan:
            planner.plan()

        path_pub.publish(planner.plan_msg())
        vel_pub.publish(planner.vel_msg())
        rate.sleep()

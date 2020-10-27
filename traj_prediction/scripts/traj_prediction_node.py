#!/usr/bin/env python

import rospy

from traj_prediction.nn_predictor import NNPredictor

from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from nav_msgs.msg import Odometry, OccupancyGrid

if __name__ == '__main__':
    predictor = NNPredictor()

    rospy.init_node('traj_prediction_node')
    rate = rospy.Rate(2)

    heightmap_sub = rospy.Subscriber('/heightmap', OccupancyGrid, predictor.handle_heightmap)
#    heightmap_sub = rospy.Subscriber('/map', OccupancyGrid, predictor.handle_heightmap)
    pose_sub = rospy.Subscriber('/car/particle_filter/inferred_pose', PoseStamped, predictor.handle_pose)
    odom_sub = rospy.Subscriber('/car/vesc/odom', Odometry, predictor.handle_odom)
    plan_sub = rospy.Subscriber('/car/planner/path', PoseArray, predictor.handle_plan)

    predict_pub = rospy.Publisher('/traj_prediction', PoseArray, queue_size=1)
    segment_pub = rospy.Publisher('/path_segment', PoseArray, queue_size=1)

    for _ in range(4):
        rate.sleep()

    while not rospy.is_shutdown():
        print(predictor.can_predict)
        if predictor.can_predict:
            predictor.predict()
            predictor.render()
            predict_pub.publish(predictor.predict_msg)
            segment_pub.publish(predictor.plan_segment_msg)
            
        rate.sleep()

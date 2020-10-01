#!/usr/bin/python

import rospy
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import Point, Pose, PoseWithCovariance, PoseWithCovarianceStamped, Quaternion
from tf.transformations import quaternion_from_euler

def run_plan(pub_init_pose, pub_controls, plan):
    init = plan.pop(0)
    send_init_pose(pub_init_pose, init)

    for c in plan:
        send_command(pub_controls, c)

def send_init_pose(pub_init_pose, init_pose):
    pose_data = init_pose.split(',')
    assert len(pose_data) == 3

    x, y, theta = [float(d) for d in pose_data]
    q = Quaternion(*quaternion_from_euler(0, 0, theta))
    point = Point(x=x, y=y)
    pose = PoseWithCovariance(pose=Pose(position=point, orientation=q))
    pub_init_pose.publish(PoseWithCovarianceStamped(pose=pose))

def send_command(pub_controls, c):
    cmd = c.split(',')
    assert len(cmd) == 2

    v, delta = [float(d) for d in cmd]
    dur = rospy.Duration(1.0)
    rate = rospy.Rate(10)
    start = rospy.Time.now()

    drive = AckermannDrive(steering_angle=delta, speed=v)

    while (rospy.Time.now() - start) < dur:
        pub_controls.publish(AckermannDriveStamped(drive=drive))
        rate.sleep()

if __name__ == '__main__':
    rospy.init_node("path_publisher")

    control_topic = rospy.get_param("~control_topic", "/car/mux/ackermann_cmd_mux/input/navigation")
    pub_controls = rospy.Publisher(control_topic, AckermannDriveStamped, queue_size=1)

    init_pose_topic = rospy.get_param("~init_pose_topic", "/initialpose")
    pub_init_pose = rospy.Publisher(init_pose_topic, PoseWithCovarianceStamped, queue_size=1)

    plan_file = rospy.get_param("~plan_file")

    with open(plan_file) as f:
        plan = f.readlines()

    rospy.sleep(1.0)
    run_plan(pub_init_pose, pub_controls, plan)

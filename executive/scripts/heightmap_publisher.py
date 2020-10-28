#!/usr/bin/env python

import rospy
import numpy as np
import pickle
import os

from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData
from geometry_msgs.msg import Point, Quaternion, Pose

def heightmap_msg(heightmap, metadata):
    """
    Build the heightmap from the given numpy file and metadata
    For now, say that each elem in occupancy grid = 1cm
    """
    msg = OccupancyGrid()
    msg.header.frame_id='/map'
    hmap_int = (heightmap*100).astype(np.uint8)
    msg.data = hmap_int.flatten()

    msg.info = MapMetaData(
        resolution=metadata['resoluion'],
        width=metadata['width'],
        height=metadata['height'],
        origin=Pose(
            position=Point(
                x=metadata['origin']['position']['x'],
                y=metadata['origin']['position']['y'],
                z=metadata['origin']['position']['z'],
            ),
            orientation=Quaternion(
                x=metadata['origin']['orientation']['x'],
                y=metadata['origin']['orientation']['y'],
                z=metadata['origin']['orientation']['z'],
                w=metadata['origin']['orientation']['w'],
            )
        )
    )
    assert len(msg.data) == msg.info.width * msg.info.height, 'BAD'
    return msg

def main():
    rospy.init_node('heightmap_publisher')

    heightmap_pub = rospy.Publisher('/heightmap', OccupancyGrid, queue_size=1)
    rate = rospy.Rate(1)

    print(os.getcwd())

    heightmap = np.load('src/executive/heightmaps/heightmap.npy')
    metadata = pickle.load(open('src/executive/heightmaps/metadata.pkl', 'rb'))

    msg = heightmap_msg(heightmap, metadata)

    while not rospy.is_shutdown():
        heightmap_pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
     main()

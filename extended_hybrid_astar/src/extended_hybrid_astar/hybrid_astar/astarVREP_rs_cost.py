import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import time as tim
from PIL import Image
import hybrid_a_star_rs_cost as has
#import scipy.ndimage
import math

def useEdges(heightmap):
    edges = cv2.Canny(np.uint8(heightmap > 0.0),0,1)
    heightmap = cv2.bitwise_and(heightmap,heightmap,mask=edges)

    return heightmap

def getObstacles(heightmap,threshold=0):
    x,y = np.where(heightmap.astype(float) > threshold)
    return x,y

def removeGoal(heightmap):
    #find goal location and remove it as an obstacle
    hmcopy = np.uint8((heightmap > 0.0))
    output = cv2.connectedComponentsWithStats(hmcopy, 8, cv2.CV_32S)

    labels = output[1]
    centroids = output[3].astype(int)
    goal = np.argmax(heightmap[centroids[:,1],centroids[:,0]])
    goaly = centroids[goal,1]
    goalx = centroids[goal,0]

    mask = np.array(labels, dtype=np.uint8)
    mask[labels == goal] = 0
    mask[labels != goal] = 1

    #take out goal
    heightmap = heightmap*mask

    return goaly,goalx,heightmap


def plan_from_VREP(heightmap,startx,starty,startyaw,goalx,goaly,goalyaw,anglemap,model,extended=True, hmap_threshold=0.):

#    goalyaw = -1*math.pi#-3.14

    anglemap = np.pad(anglemap,pad_width=1, mode='constant',constant_values=1 )

    heightmap = np.pad(heightmap,pad_width=1, mode='constant',constant_values=1 )

    x,y = getObstacles(heightmap, threshold=hmap_threshold)
    # np.save('x.npy',x)
    # np.save('y.npy',y)

    rx,ry,ryaw, locs,directions,vel,ax,ay = has.plan(startx,starty,startyaw,x,y,goalx,goaly,goalyaw,heightmap,anglemap,model,extended)
    #rx,ry,ryaw, locs,directions,vel = has.plan(startx,starty,startyaw,x,y,goalx,goaly,goalyaw,heightmap,anglemap,extended)

    return rx,ry,ryaw,vel,ax,ay

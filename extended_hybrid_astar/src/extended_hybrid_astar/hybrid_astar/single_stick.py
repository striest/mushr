import numpy as np

import sys
sys.path.append("../")

from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
from pyrep import PyRep
from robotnik import nik
from nik_path_base import NikPathBase
import cv2
from math import cos, sin, pi
import matplotlib.pyplot as plt
import astarVREP as ast

def draw_rectangle(img, x, y, w, l, angle, height):
        # import ipdb;ipdb.set_trace()
        b = cos(angle) * 0.5
        a = sin(angle) * 0.5
        pt0 = (int(x - a * l - b * w),
               int(y + b * l - a * w))
        pt1 = (int(x + a * l - b * w),
               int(y - b * l - a * w))
        pt2 = (int(2 * x - pt0[0]), int(2 * y - pt0[1])) # (int(x + a * l + b * w), int(y - b * l + a * w))
        pt3 = (int(2 * x - pt1[0]), int(2 * y - pt1[1]))
        cv2.fillConvexPoly(img, np.array([pt0, pt1, pt2, pt3]), color=[height])
        return img

if __name__ == "__main__":
    # Unit test
    pr = PyRep()
    pr.launch("scene/blank_robotnik.ttt", headless=False)
    pr.start()

    # env range is 5*5
    pos_min = [-2, -1.3, 0]
    pos_max = [1.3, 2, 0]
    obstacle_num=1



    # agent
    agent = nik('Robotnik')

    # path
    target = Shape.create(type=PrimitiveShape.SPHERE,
                      size=[0.05, 0.05, 0.05],
                      color=[1.0, 0.1, 0.1],
                      static=True, respondable=False)
    position_min, position_max = [-0.5, 1, 0.1], [0.5, 1.5, 0.1]

    starting_pose = agent.get_2d_pose()
    agent.set_motor_locked_at_zero_velocity(True)

    for i in range(1):
        heightmap = np.zeros((450, 450), dtype=np.float32)
        blocks = []
        for i in range(obstacle_num):
            # stick_size = [0.05, 1, np.random.uniform(0.01, 0.2)]
            stick_size = [0.05, 1, .1]
            ob_pos = [0,0,0]
            ob_or = [0, 0, 0]

            obstacle = Shape.create(type=PrimitiveShape.CUBOID, size=stick_size,
                                    color=[stick_size[2]*10]*3, orientation=ob_or,
                                    static=True, position=ob_pos, respondable=True)
            obstacle.set_collidable(True)
            blocks.append(obstacle)

            height = stick_size[2]
            print('height = '+ str(height))
            x, y = int((ob_pos[0]+2.1)*100),int((ob_pos[1]+2.1)*100)
            print((x,y))
            # print((x,y,height))
            heightmap = draw_rectangle(heightmap, x, y,
                                        int(stick_size[0]*100), int(stick_size[1]*100),
                                        ob_or[2], height)

        starting_pose = [1.5,0,starting_pose[2]]
        # new_pose = starting_pose
        # print(new_pose)
        agent.set_2d_pose(starting_pose)

        # Get a random position within a cuboid and set the target position
        target_pos = [-1.5,0,0]
        target.set_position(target_pos)

        path_base = NikPathBase(agent)
        path = path_base.get_nonlinear_path(position=target_pos, angle=0)
        # path = path_base.get_linear_path(position=target_pos, angle=0)

        anglemap = heightmap.copy()

        starty = int((starting_pose[0]+2.1)*100)
        startx =int((starting_pose[1]+2.1)*100)
        startyaw = 0-3.14
        goaly = int((target_pos[0]+2.1)*100)
        goalx = int((target_pos[1]+2.1)*100)

        rx,ry,ryaw = ast.plan_from_VREP(heightmap,startx,starty,startyaw,goalx,goaly,anglemap,extended=True)


        # points = path._path_points

        newpoints = []
        for x,y in zip(rx,ry):
            newx = x/100 - 2.1
            newy = y/100 - 2.1
            newpoints.append([newx,newy,1,1])

        path._path_points = newpoints

        path.visualize()
        done = False

        fx = np.asarray([])
        fy = np.asarray([])
        fyaw = np.asarray([])
        fvelocity = np.asarray([])

        cur_step = 0
        while not done and cur_step <= 6000:
            cur_pose = agent.get_2d_pose()
            done = path_base.step(path)
            # agent.set_velocity(10,10,10,10)
            pr.step()
            vel = agent.get_velocity()

            fx = np.append(fx,(cur_pose[0]+2.1)*100)
            fy = np.append(fy,(cur_pose[1]+2.1)*100)
            fyaw = np.append(fyaw,cur_pose[2])
            fvelocity = np.append(fvelocity,vel)
            cur_step += 1

        path.clear_visualization()
        # for block in blocks:
        #     del block

        print('Reached target %d!' % i)
        plt.imshow(heightmap)
        plt.plot(starty,startx,'.g')
        plt.plot(goaly,goalx,'.b')
        plt.plot(rx,ry,'-r')
        plt.plot(fx,fy,'-k')
        plt.gca().invert_yaxis()
        # plt.show()
        plt.savefig('newfig.png')

        cv2.imwrite('temp.png',heightmap)

    pr.stop()
    pr.shutdown()

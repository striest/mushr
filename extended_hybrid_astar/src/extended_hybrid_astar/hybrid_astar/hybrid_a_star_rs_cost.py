"""
Hybrid A* path planning
author: Zheng Zh (@Zhengzh)
"""

import heapq
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import torch
import skimage.transform
from numpy import sin, cos, pi
from skimage.transform import ProjectiveTransform

import time

try:
    # from a_star_heuristic import dp_planning
    from dynamic_programming_heuristic import calc_distance_heuristic
    import reeds_shepp_path_planning as rs
    from car import move, check_car_collision, MAX_STEER, WB, plot_car
except Exception:
    raise

MODEL = None
MAXTIME = 30
cur_time = 0
MIN_BEST_PATH = None
MIN_BEST_COST = None
MIN_BEST_VEL = None
MIN_COST = 2400
USE_MIN_COST = True
SAVE_INDEX = 0
ACTUAL_BEST_PATH = None

XY_GRID_RESOLUTION = 20.0 #4.0 #5.0  # [m]
YAW_GRID_RESOLUTION = np.deg2rad(20.0)#np.deg2rad(5.0)  # [rad]
MOTION_RESOLUTION = 1.5  # [m] path interpolate resolution #originally .5
N_STEER = 50  # number of steer command
VR = 10.0  # robot radius


SB_COST = 10000.0  # switch back penalty cost
BACK_COST = 50.0  # backward penalty cost

# SB_COST = 10000000000000000.0  # switch back penalty cost
# BACK_COST = 5000000000000.0  # backward penalty cost
# SB_COST = 10000000.0  # switch back penalty cost
# BACK_COST = 50000000.0  # backward penalty cost
STEER_CHANGE_COST = 5.0  # steer angle change penalty cost
STEER_COST = 1.0  # steer angle change penalty cost
H_COST = 5.0  # Heuristic cost

show_animation = False

h_map = []
a_map = []
extend = True

class Node:

    def __init__(self, x_ind, y_ind, yaw_ind, direction,
                 x_list, y_list, yaw_list, directions,
                 steer=0.0, parent_index=None, cost=None):

        reso = int(XY_GRID_RESOLUTION)
        height = h_map[int(y_ind*reso)][int(x_ind*reso)]
        # height = 0

        cur_yaw = yaw_ind*-1 + math.pi/2
        if cur_yaw < 0:
            cur_yaw += 3.14
        angle_diff = 0

        # print('test')
        # print(x_ind)
        # print(y_ind)
        # plt.imshow(h_map)
        # plt.plot(x_list,y_list)
        # plt.savefig('test.png')
        # print((x_ind,y_ind))
        # print((x_list,y_list))
        if extend:

            if height == 0.0:
                buffer = int(VR/1)
                window = h_map[(int(y_ind)*reso - buffer):(int(y_ind)*reso + buffer),(int(x_ind)*reso - buffer):(int(x_ind)*reso + buffer)]
                # window = h_map[(int(x_ind)*reso - buffer):(int(x_ind)*reso + buffer),(int(y_ind)*reso - buffer):(int(y_ind)*reso + buffer)]
                height = np.amax(window)
                if height > 0.0:
                    height_index = np.unravel_index(np.argmax(window),window.shape)

                    new_y = int(y_ind)*reso - buffer + height_index[0]
                    new_x = int(x_ind)*reso - buffer + height_index[1]
                    # print((new_y,new_x))
                    # print(a_map[new_y][new_x])
                    # raw_input("Press Enter to start path following...")

                    obs_angle = a_map[height_index[0]][height_index[1]]
                    angle_diff = abs(np.cos(obs_angle - cur_yaw))
                else:
                    angle_diff = 0
            elif height > 0.0:
                #plt.imshow(abs(np.cos(a_map - cur_yaw)))
                #plt.show()
                # buffer = int(VR/8)
                # window = a_map[(int(y_ind)*reso - buffer):(int(y_ind)*reso + buffer),(int(x_ind)*reso - buffer):(int(x_ind)*reso + buffer)]
                # if window.size != 0:
                #     obs_angle = np.amax(window)
                #     angle_diff = abs(np.cos(obs_angle - cur_yaw))
                obs_angle = a_map[int(y_ind*reso)][int(x_ind*reso)]
                angle_diff = abs(np.cos(obs_angle - cur_yaw))
        else:
            height = 0
            angle_diff = 0

        self.x_index = x_ind
        self.y_index = y_ind
        self.yaw_index = yaw_ind
        self.direction = direction
        self.x_list = x_list
        self.y_list = y_list
        self.yaw_list = yaw_list
        self.directions = directions
        self.steer = steer
        self.parent_index = parent_index
        if cost == None:
            self.cost = cost
        else:
            self.cost = cost + height*1000000# + angle_diff*8000#height*10000 + angle_diff*8000
        # print(height)
        if height > 0:
            #print((cost, height, angle_diff))
            pass


class Path:

    def __init__(self, x_list, y_list, yaw_list, direction_list, cost):
        self.x_list = x_list
        self.y_list = y_list
        self.yaw_list = yaw_list
        self.direction_list = direction_list
        self.cost = cost


class Config:

    def __init__(self, ox, oy, xy_resolution, yaw_resolution):
        min_x_m = min(ox)
        min_y_m = min(oy)
        max_x_m = max(ox)
        max_y_m = max(oy)

        # ox.append(min_x_m)
        # oy.append(min_y_m)
        # ox.append(max_x_m)
        # oy.append(max_y_m)

        np.append(ox,min_x_m)
        np.append(ox,max_x_m)
        np.append(oy,min_y_m)
        np.append(oy,max_y_m)

        self.min_x = round(min_x_m / xy_resolution)
        self.min_y = round(min_y_m / xy_resolution)
        self.max_x = round(max_x_m / xy_resolution)
        self.max_y = round(max_y_m / xy_resolution)

        self.x_w = round(self.max_x - self.min_x)
        self.y_w = round(self.max_y - self.min_y)

        self.min_yaw = round(- math.pi / yaw_resolution) - 1
        self.max_yaw = round(math.pi / yaw_resolution)
        self.yaw_w = round(self.max_yaw - self.min_yaw)


def calc_motion_inputs():
    for steer in np.concatenate((np.linspace(-MAX_STEER, MAX_STEER,
                                             N_STEER), [0.0])):
        for d in [1, -1]:
            yield [steer, d]


def get_neighbors(current, config, ox, oy, kd_tree):
    for steer, d in calc_motion_inputs():
        node = calc_next_node(current, steer, d, config, ox, oy, kd_tree)
        if node and verify_index(node, config):
            yield node


def calc_next_node(current, steer, direction, config, ox, oy, kd_tree):
    x, y, yaw = current.x_list[-1], current.y_list[-1], current.yaw_list[-1]

    arc_l = XY_GRID_RESOLUTION * 1.5
    x_list, y_list, yaw_list = [], [], []
    for _ in np.arange(0, arc_l, MOTION_RESOLUTION):
        x, y, yaw = move(x, y, yaw, MOTION_RESOLUTION * direction, steer)
        x_list.append(x)
        y_list.append(y)
        yaw_list.append(yaw)

    if not check_car_collision(x_list, y_list, yaw_list, ox, oy, kd_tree):
        return None

    d = direction == 1
    x_ind = round(x / XY_GRID_RESOLUTION)
    y_ind = round(y / XY_GRID_RESOLUTION)
    yaw_ind = round(yaw / YAW_GRID_RESOLUTION)

    added_cost = 0.0

    if d != current.direction:
        added_cost += SB_COST

    # steer penalty
    added_cost += STEER_COST * abs(steer)

    # steer change penalty
    added_cost += STEER_CHANGE_COST * abs(current.steer - steer)

    cost = current.cost + added_cost + arc_l


    node = Node(x_ind, y_ind, yaw_ind, d, x_list,
                y_list, yaw_list, [d],
                parent_index=calc_index(current, config),
                cost=cost, steer=steer)

    return node


def is_same_grid(n1, n2):
    if n1.x_index == n2.x_index \
            and n1.y_index == n2.y_index \
            and n1.yaw_index == n2.yaw_index:
        return True
    return False


def analytic_expansion(current, goal, ox, oy, kd_tree,closedList,c):
    start_x = current.x_list[-1]
    start_y = current.y_list[-1]
    start_yaw = current.yaw_list[-1]

    goal_x = goal.x_list[-1]
    goal_y = goal.y_list[-1]
    goal_yaw = goal.yaw_list[-1]

    max_curvature = math.tan(MAX_STEER) / WB
    paths = rs.calc_paths(start_x, start_y, start_yaw,
                          goal_x, goal_y, goal_yaw,
                          max_curvature, step_size=MOTION_RESOLUTION)

    #print(len(paths))
    if not paths:
        return None

    best_path, best = None, None
    vel = 0
    actual_path = None

    for path in paths:
        if check_car_collision(path.x, path.y, path.yaw, ox, oy, kd_tree):
            cost = calc_rs_path_cost(path) #+cost from heightmap
            #print((cost,start_x,start_y,path.x[0],path.y[0],current.x_list[0],current.y_list[0]))
            #path is getting modified...?
            f_x = path.x[1:]
            f_y = path.y[1:]
            f_yaw = path.yaw[1:]

            f_cost = current.cost + cost
            f_parent_index = calc_index(current, c) #current.parent_index

            fd = []
            for d in path.directions[1:]:
                fd.append(d >= 0)

            f_steer = 0.0
            f_path = Node(current.x_index, current.y_index, current.yaw_index,
                          current.direction, f_x, f_y, f_yaw, fd,
                          cost=f_cost, parent_index=f_parent_index, steer=f_steer)
            temp_path = get_path(closedList, f_path)

            # print(len(temp_path.x_list))
            # plt.plot(temp_path.x_list,temp_path.y_list,'.b')
            # plt.ylim(0,450)
            # plt.xlim(0,450)
            # global SAVE_INDEX
            # plt.savefig('debug/' + str(SAVE_INDEX)+ '.png')
            # plt.clf()
            # SAVE_INDEX +=1

            m_cost,vel = infer_cost_from_model(temp_path)
            cost += m_cost
            #cost += infer_cost_height(temp_path)
            #just to validate that temp_path starts at start and ends at goal before running cost evaluation
            #print((cost,start_x,start_y,temp_path.x_list[0],temp_path.y_list[0],temp_path.x_list[-1],temp_path.y_list[-1]))
            #2400
            # if cost < 1000 and (not best or best > cost):
            #     print('\n\n\n\n')
            #     best = cost
            #     best_path = path
            if not best or best > cost:
                #print('\n\n\n\n')
                best = cost
                best_path = path
                actual_path = temp_path

    global MIN_BEST_COST,MIN_BEST_PATH, MIN_BEST_VEL, ACTUAL_BEST_PATH

    if best is not None:
        if MIN_BEST_COST is None:
            MIN_BEST_COST = best
            MIN_BEST_PATH = best_path
            MIN_BEST_VEL = vel
            ACTUAL_BEST_PATH = actual_path
        elif MIN_BEST_COST > best:
            MIN_BEST_COST = best
            MIN_BEST_PATH = best_path
            MIN_BEST_VEL = vel
            ACTUAL_BEST_PATH = actual_path

#    time_diff = time.perf_counter() - cur_time
    time_diff = time.clock() - cur_time
    #print(time_diff)
    if time_diff > MAXTIME:
        #print(type(MIN_BEST_PATH))
        return MIN_BEST_PATH, MIN_BEST_VEL
    # print(cost)
    if best is not None and best < MIN_COST and USE_MIN_COST:
        return best_path, vel

    return None,None


def infer_cost_height(path):
    cost = 0

    xlist = path.x_list
    ylist = path.y_list
    yawlist = path.yaw_list

    reso = int(XY_GRID_RESOLUTION)
    for x,y,yaw in zip(xlist,ylist,yawlist):
        # print(h_map.shape)
        # print((x,y))
        height = h_map[int(y)][int(x)]
        if height == 0.0:
            buffer = int(VR/1)
            window = h_map[(int(y) - buffer):(int(y) + buffer),(int(x)- buffer):(int(x) + buffer)]
            # window = h_map[(int(x_ind)*reso - buffer):(int(x_ind)*reso + buffer),(int(y_ind)*reso - buffer):(int(y_ind)*reso + buffer)]
            height = np.amax(window)
        cost += height*1000

    return cost

def get_cropped_heightmap(state,heightmap):
        """
        Gets the correct crop of the heightmap
        """
        map_x, map_y = state[:2]
        th = state[2]
        windowsize = 100

        #OK, to properly get the heightmap, we need to 1. translate to vehicle origin. 2. Rotate by theta 3. translate by -windowsize/2 to center
        HTM_trans = np.array([[1., 0., map_x], [0., 1., map_y], [0., 0., 1.]])
        HTM_rot = np.array([[cos(th), -sin(th), 0.], [sin(th), cos(th), 0.], [0., 0., 1.]])
        HTM_center = np.array([[1., 0., -windowsize//2], [0., 1., -windowsize//2], [0., 0., 1.]])
        HTM = np.matmul(HTM_trans, np.matmul(HTM_rot, HTM_center))
        heightmap_tr = skimage.transform.warp(heightmap, ProjectiveTransform(matrix=HTM))
        heightmap_out = heightmap_tr[:windowsize, :windowsize]

        return heightmap_out
def transform_traj(traj, ego_HTM, th):
        """
        Rotate/translate traj to be in the coordinate frame of state.

        """
        traj_pos = np.concatenate([traj[:, :2], np.ones([len(traj), 1])], axis=1)
        traj_pos = np.matmul(ego_HTM, traj_pos.transpose()).transpose()
        traj_th = traj[:, 2] - th
        traj_pos[:, 2] = traj_th
        return traj_pos

def calculateDistance(point1,point2):

     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
     return dist

def calculateDistance(point1,point2):
    x1,y1 = point1[0],point1[1]
    x2,y2 = point2[0],point2[1]
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

def pathCompare(target_traj,prediction):
    cost = 0
    for i in range(10):
        targetx = target_traj[i,0]
        targety = target_traj[i,1]
        target = [targetx,targety]

        prediction_target = [prediction[i,0],prediction[i,1]]

        cost += calculateDistance(target,prediction_target)**2
    return cost

def infer_cost_from_model(path):
    """

    """
    #at some point, eval_reso should have relationship with plan_step number as well as path length
    eval_reso = 30
    num_steps = 30
    skip_a = num_steps//10

    xlist = path.x_list
    ylist = path.y_list
    yawlist = path.yaw_list

    vel_opts = [.03,.04,.06,.08,.09]
    vel_opts = [0.3 * 5, 0.4 * 5, 0.5 * 5, 0.6 * 5, 0.7 * 5]

    best_cost = 10000000000000000000000000000
    best_vel = .03
    for vel in vel_opts:
        cost = 0

        for i in range(len(xlist)//eval_reso):
        #need to calculate state, next 20 steps plan, heightmap
        #need to calculate predictions at different velocities
        #state - need to validate yaw - it's wrong for reversed, need to reflect in state

            state_np = np.array([xlist[i*eval_reso], ylist[i*eval_reso], yawlist[i*eval_reso], vel]).astype(np.float32)

            #next 20 steps of plan
            if((i*eval_reso + num_steps) < len(xlist)):
                plan_traj = np.zeros([10,3]).astype(np.float32)
                # plan_traj[:,2] = yawlist[i*eval_reso:i*eval_reso + num_steps]
                # plan_traj[:,0] = xlist[i*eval_reso:i*eval_reso + num_steps]
                # plan_traj[:,1] = ylist[i*eval_reso:i*eval_reso + num_steps]
                plan_traj[:,2] = yawlist[i*eval_reso:(i*eval_reso + num_steps):skip_a]
                plan_traj[:,0] = xlist[i*eval_reso:(i*eval_reso + num_steps):skip_a]
                plan_traj[:,1] = ylist[i*eval_reso:(i*eval_reso + num_steps):skip_a]
            else:
                break



            x, y, th = state_np[:3]
            ego_HTM_tr = np.array([[1., 0., -x],[0., 1., -y],[0., 0., 1.]])
            ego_HTM_rot = np.array([[cos(-th), -sin(-th), 0.], [sin(-th), cos(-th), 0.], [0., 0., 1.]])
            ego_HTM = np.matmul(ego_HTM_rot, ego_HTM_tr)
            plan_np = transform_traj(plan_traj, ego_HTM, th)



            heightmap = h_map
            heightmap_np = get_cropped_heightmap(state_np,heightmap)

            state_np_new = np.array([0., 0., 0., state_np[3]])

            heightmap_torch = torch.tensor(skimage.transform.resize(heightmap_np, [24, 24])).unsqueeze(0).unsqueeze(0).float()
            plan_torch = torch.tensor(plan_np).unsqueeze(0).float()
            state_torch = torch.tensor(state_np_new).unsqueeze(0).unsqueeze(0).float()

            with torch.no_grad():
                preds_torch = MODEL.forward({'state':state_torch, 'heightmap':heightmap_torch, 'traj_plan':plan_torch})


            prediction = preds_torch.squeeze().cpu().numpy()
            ego_HTM = np.array([[cos(th), -sin(th), x], [sin(th), cos(th), y], [0., 0., 1.]])
            prediction = transform_traj(prediction, ego_HTM, -th)

            cost += pathCompare(plan_traj,prediction)

            # VREP Doesn't like this
            # plt.imshow(heightmap,origin='lower')
            # plt.plot(plan_traj[:,0][0],plan_traj[:,1][0],'.r')
            # plt.plot(plan_traj[:,0][-1],plan_traj[:,1][-1],'.b')
            #
            # plt.plot(plan_traj[:,0],plan_traj[:,1])
            # plt.plot(prediction[:,0],prediction[:,1])
            # plt.plot(prediction[-1,0],prediction[-1,1],'.g')
            # plt.savefig('Intermediate/'+ str(i) + '.png')
            # plt.clf()

        if cost < best_cost:
            best_cost = cost
            best_vel = vel


    return cost,vel



def update_node_with_analytic_expansion(current, goal,
                                        c, ox, oy, kd_tree,closedList):
    #Tries to generate path from current to goal using reedshepp
    #IF there is one without collision, that's considered the best path and it quits
    path,vel = analytic_expansion(current, goal, ox, oy, kd_tree,closedList,c)

    #look for code here that contains whole path
    if path:
        if show_animation:
            plt.plot(path.x, path.y)
        f_x = path.x[1:]
        f_y = path.y[1:]
        f_yaw = path.yaw[1:]

        f_cost = current.cost + calc_rs_path_cost(path)
        f_parent_index = calc_index(current, c) #try current parent

        fd = []
        for d in path.directions[1:]:
            fd.append(d >= 0)

        f_steer = 0.0
        f_path = Node(current.x_index, current.y_index, current.yaw_index,
                      current.direction, f_x, f_y, f_yaw, fd,
                      cost=f_cost, parent_index=f_parent_index, steer=f_steer)
        #print((f_x,f_y))
        return True, f_path,vel

    return False, None,vel


def calc_rs_path_cost(reed_shepp_path):
    cost = 0.0
    for length in reed_shepp_path.lengths:
        if length >= 0:  # forward
            cost += length
        else:  # back
            cost += abs(length) * BACK_COST

    # switch back penalty
    for i in range(len(reed_shepp_path.lengths) - 1):
        # switch back
        if reed_shepp_path.lengths[i] * reed_shepp_path.lengths[i + 1] < 0.0:
            cost += SB_COST

    # steer penalty
    for course_type in reed_shepp_path.ctypes:
        if course_type != "S":  # curve
            cost += STEER_COST * abs(MAX_STEER)

    # ==steer change penalty
    # calc steer profile
    n_ctypes = len(reed_shepp_path.ctypes)
    u_list = [0.0] * n_ctypes
    for i in range(n_ctypes):
        if reed_shepp_path.ctypes[i] == "R":
            u_list[i] = - MAX_STEER
        elif reed_shepp_path.ctypes[i] == "L":
            u_list[i] = MAX_STEER

    for i in range(len(reed_shepp_path.ctypes) - 1):
        cost += STEER_CHANGE_COST * abs(u_list[i + 1] - u_list[i])

    return cost


def hybrid_a_star_planning(start, goal, ox, oy, xy_resolution, yaw_resolution):
    """
    start: start node
    goal: goal node
    ox: x position list of Obstacles [m]
    oy: y position list of Obstacles [m]
    xy_resolution: grid resolution [m]
    yaw_resolution: yaw angle resolution [rad]
    """

    plt.show(block=False)
    plt.clf()
    plt.scatter(ox, oy, s=1.0)
    plt.scatter(start[0], start[1], c='r', marker='x')
    plt.scatter(goal[0], goal[1], c='g', marker='x')
    plt.pause(1e-2)


    start[2], goal[2] = rs.pi_2_pi(start[2]), rs.pi_2_pi(goal[2])
    tox, toy = ox[:], oy[:]

    obstacle_kd_tree = cKDTree(np.vstack((tox, toy)).T)

    config = Config(tox, toy, xy_resolution, yaw_resolution)

    start_node = Node(round(start[0] / xy_resolution),
                      round(start[1] / xy_resolution),
                      round(start[2] / yaw_resolution), True,
                      [start[0]], [start[1]], [start[2]], [True], cost=0)
    goal_node = Node(round(goal[0] / xy_resolution),
                     round(goal[1] / xy_resolution),
                     round(goal[2] / yaw_resolution), True,
                     [goal[0]], [goal[1]], [goal[2]], [True])

    openList, closedList = {}, {}

    h_dp = calc_distance_heuristic(
        goal_node.x_list[-1], goal_node.y_list[-1],
        ox, oy, xy_resolution, VR)

    pq = []
    openList[calc_index(start_node, config)] = start_node
    heapq.heappush(pq, (calc_cost(start_node, h_dp, config),
                        calc_index(start_node, config)))
    final_path = None

    #print(len(openList))

    global cur_time
#    cur_time = time.perf_counter()
    cur_time = time.clock()
    while True:
        if not openList:
            print("Error: Cannot find path, No open set")
            return [], [], []

        cost, c_id = heapq.heappop(pq)
        if c_id in openList:
            current = openList.pop(c_id)
            closedList[c_id] = current
        else:
            continue

        if show_animation:  # pragma: no cover
            plt.plot(current.x_list[-1], current.y_list[-1], "xc")
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            if len(closedList.keys()) % 10 == 0:
                plt.pause(0.001)

        is_updated, final_path,vel = update_node_with_analytic_expansion(
            current, goal_node, config, ox, oy, obstacle_kd_tree,closedList)


        if is_updated:
            print("path found")
            break

        #these don't get expanded when path is found
        for neighbor in get_neighbors(current, config, ox, oy,
                                      obstacle_kd_tree):
            neighbor_index = calc_index(neighbor, config)
            if neighbor_index in closedList:
                continue
            if neighbor not in openList \
                    or openList[neighbor_index].cost > neighbor.cost:
                heapq.heappush(
                    pq, (calc_cost(neighbor, h_dp, config),
                         neighbor_index))
                openList[neighbor_index] = neighbor

    path = get_final_path(closedList, final_path)
    return path,vel


def calc_cost(n, h_dp, c):
    ind = (n.y_index - c.min_y) * c.x_w + (n.x_index - c.min_x)
    if ind not in h_dp:
        return n.cost + 999999999  # collision cost
    return n.cost + H_COST * h_dp[ind].cost


def get_path(closed, goal_node):
    reversed_x, reversed_y, reversed_yaw = \
        list(reversed(goal_node.x_list)), list(reversed(goal_node.y_list)), \
        list(reversed(goal_node.yaw_list))
    direction = list(reversed(goal_node.directions))
    nid = goal_node.parent_index
    final_cost = goal_node.cost

    while nid:
        n = closed[nid]
        reversed_x.extend(list(reversed(n.x_list)))
        reversed_y.extend(list(reversed(n.y_list)))
        reversed_yaw.extend(list(reversed(n.yaw_list)))
        direction.extend(list(reversed(n.directions)))

        nid = n.parent_index

    reversed_x = list(reversed(reversed_x))
    reversed_y = list(reversed(reversed_y))
    reversed_yaw = list(reversed(reversed_yaw))
    direction = list(reversed(direction))

    # adjust first direction
    direction[0] = direction[1]

    path = Path(reversed_x, reversed_y, reversed_yaw, direction, final_cost)
    return path

def get_final_path(closed, goal_node):
    reversed_x, reversed_y, reversed_yaw = \
        list(reversed(goal_node.x_list)), list(reversed(goal_node.y_list)), \
        list(reversed(goal_node.yaw_list))
    direction = list(reversed(goal_node.directions))
    nid = goal_node.parent_index
    final_cost = goal_node.cost

    while nid:
        n = closed[nid]
        reversed_x.extend(list(reversed(n.x_list)))
        reversed_y.extend(list(reversed(n.y_list)))
        reversed_yaw.extend(list(reversed(n.yaw_list)))
        direction.extend(list(reversed(n.directions)))

        nid = n.parent_index

    reversed_x = list(reversed(reversed_x))
    reversed_y = list(reversed(reversed_y))
    reversed_yaw = list(reversed(reversed_yaw))
    direction = list(reversed(direction))

    # adjust first direction
    direction[0] = direction[1]

    path = Path(reversed_x, reversed_y, reversed_yaw, direction, final_cost)
    return path


def verify_index(node, c):
    x_ind, y_ind = node.x_index, node.y_index
    if c.min_x <= x_ind <= c.max_x and c.min_y <= y_ind <= c.max_y:
        return True

    return False


def calc_index(node, c):
    ind = (node.yaw_index - c.min_yaw) * c.x_w * c.y_w + \
          (node.y_index - c.min_y) * c.x_w + (node.x_index - c.min_x)

    if ind <= 0:
        print("Error(calc_index):", ind)

    return ind

def plan(startx,starty,startyaw,x,y,goalx,goaly,goalyaw,heightmap,anglemap,model,extended=True):
    print("Start Hybrid A* planning")

    print(heightmap.shape)
    global h_map
    h_map = heightmap
    global a_map
    a_map = anglemap
    global extend
    extend = extended
    global MODEL
    MODEL = model

    ox, oy = y,x

    # Set Initial parameters
    start = [starty, startx, startyaw]
    goal = [goaly, goalx, goalyaw]

    print("start : ", start)
    print("goal : ", goal)

    if show_animation:
        plt.plot(ox, oy, ".k")
        rs.plot_arrow(start[0], start[1], start[2], fc='g')
        rs.plot_arrow(goal[0], goal[1], goal[2])

        plt.grid(True)
        plt.axis("equal")

    path,vel = hybrid_a_star_planning(
        start, goal, ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)

    x = path.x_list
    y = path.y_list
    yaw = path.yaw_list
    directions = path.direction_list


    coords = np.column_stack((np.array(x),np.array(y)))
    grad = np.gradient(coords,axis=0)

    # for i in range(0,len(x)):
    #     print(str(i) + '  ' +  str(x[i]) + '  ' + str(y[i])  + '  ' + str(grad[i]))

    ###OPTIMIZE THIS LATER IF NEEDED
    locs = [0]
    for i in range(0,len(grad)):
        if abs(grad[i,0]) < .1:
            if abs(grad[i,1]) < .1:
                locs.append(i)
    locs.append(len(x) - 1)

    if show_animation:
        for i_x, i_y, i_yaw in zip(x, y, yaw):
            plt.cla()
            plt.gca().invert_yaxis()
            plt.plot(ox, oy, ".k")
            plt.plot(x, y, "-r", label="Hybrid A* path")
            plt.grid(True)
            plt.axis("equal")
            plt.imshow(heightmap)
            plot_car(i_x, i_y, i_yaw)
            plt.pause(0.00001)

    # plt.show()
    print(__file__ + " done!!")
    return x,y,yaw,locs,directions,vel,ACTUAL_BEST_PATH.x_list,ACTUAL_BEST_PATH.y_list

def main():
    print("Start Hybrid A* planning")

    ox, oy = [], []

    for i in range(60):
        ox.append(i)
        oy.append(0.0)
    for i in range(60):
        ox.append(60.0)
        oy.append(i)
    for i in range(61):
        ox.append(i)
        oy.append(60.0)
    for i in range(61):
        ox.append(0.0)
        oy.append(i)
    for i in range(40):
        ox.append(20.0)
        oy.append(i)
    for i in range(40):
        ox.append(40.0)
        oy.append(60.0 - i)

    # Set Initial parameters
    start = [10.0, 10.0, np.deg2rad(90.0)]
    goal = [50.0, 50.0, np.deg2rad(-90.0)]

    print("start : ", start)
    print("goal : ", goal)

    if show_animation:
        plt.plot(ox, oy, ".k")
        rs.plot_arrow(start[0], start[1], start[2], fc='g')
        rs.plot_arrow(goal[0], goal[1], goal[2])

        plt.grid(True)
        plt.axis("equal")

    path = hybrid_a_star_planning(
        start, goal, ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)

    x = path.x_list
    y = path.y_list
    yaw = path.yaw_list

    if show_animation:
        for i_x, i_y, i_yaw in zip(x, y, yaw):
            plt.cla()
            plt.plot(ox, oy, ".k")
            plt.plot(x, y, "-r", label="Hybrid A* path")
            plt.grid(True)
            plt.axis("equal")
            plot_car(i_x, i_y, i_yaw)
            plt.pause(0.0001)

    print(__file__ + " done!!")


if __name__ == '__main__':
    main()

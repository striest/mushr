from pyrep.backend import sim, utils
from pyrep.objects.dummy import Dummy
from pyrep.const import ConfigurationPathAlgorithms as Algos
from pyrep.const import PYREP_SCRIPT_TYPE
from pyrep.robots.configuration_paths.nonholonomic_configuration_path import (
    NonHolonomicConfigurationPath)
from pyrep.errors import ConfigurationPathError
from pyrep.backend._sim_cffi import ffi, lib
from math import sqrt
from math import atan2
from math import cos
from math import sin
from typing import List
from robotnik import nik


class NikPathBase(nik):
    """A path expressed in joint configuration space.

    Paths are retrieved from an :py:class:`Mobile`, and are associated with the
    mobile base that generated the path.

    This class is used for executing motion along a path via the
    _get_base_actuation function employing a proportional controller.
    """
    def __init__(self, robot):
        super(NikPathBase, self).__init__(robot.get_object_name(robot._handle))
        # Motion planning handles
        self.intermediate_target_base = Dummy.create()
        self.target_base = Dummy.create()
        self._path_done = False

        self._collision_collection = self.create_collection('Collection'.encode('ascii'))

        # Robot parameters and handle
        self.target_z = self.target_base.get_position()[-1]

        # Make sure dummies are orphan if loaded with ttm
        self.intermediate_target_base.set_parent(None)
        self.target_base.set_parent(None)

        # path para
        self.cummulative_error = 0
        self.prev_error = 0

        # PID controller values.
        self.Kp = 1.0
        self.Ki = 0.01
        self.Kd = 0.1
        self.desired_velocity = 0.05

    def check_collision_linear_path(self,path):
        """Check for collision on a linear path from start to goal

        :param path: A list containing start and goal as [x,y,yaw]
        :return: A bool, True if collision was detected
        """
        start = path[0]
        end = path[1]

        m = (end[1] - start[1])/(end[0] - start[0])
        b = start[1] - m * start[0]
        x_range = [start[0],end[0]]
        x_span = start[0] - end[0]

        incr = round(abs(x_span)/50, 3)
        if x_range[1] < x_range[0]:
            incr = - incr

        x = x_range[0]
        for k in range(50):
            x += incr
            y = m * x + b
            self.set_2d_pose([x,y,start[-1] if k < 46 else end[-1]])
            status_collision = self.assess_collision()
            if status_collision == True:
                break

        return status_collision

    def assess_collision(self):
        """Silent detection of the robot base with all other entities present in the scene.

        :return: True if collision is detected
        """
        return sim.simCheckCollision(self._collision_collection,
                                     sim.sim_handle_all) == 1

    def create_collection(self,name):
        return lib.simCreateCollection(name,0)

    def get_linear_path(self, position: List[float],
                        angle=0) -> NonHolonomicConfigurationPath:
        """Initialize linear path and check for collision along it.

        Must specify either rotation in euler or quaternions, but not both!

        :param position: The x, y position of the target.
        :param angle: The z orientation of the target (in radians).
        :raises: ConfigurationPathError if no path could be created.

        :return: A linear path in the 2d space.
        """
        position_base = self.get_position()
        angle_base = self.get_orientation()[-1]

        self.target_base.set_position(
            [position[0], position[1], self.target_z])
        self.target_base.set_orientation([0, 0, angle])
        self.intermediate_target_base.set_position(
            [position[0], position[1], self.target_z])
        self.intermediate_target_base.set_orientation([0, 0, angle])

        path = [[position_base[0], position_base[1], angle_base],
                [position[0], position[1], angle]]

        if self.check_collision_linear_path(path):
            raise ConfigurationPathError(
                'Could not create path. '
                'An object was detected on the linear path.')

        return NonHolonomicConfigurationPath(self, path)

    def get_nonlinear_path(self, position: List[float],
                           angle=0,
                           boundaries=2,
                           path_pts=600,
                           ignore_collisions=False,
                           algorithm=Algos.RRTConnect
                           ) -> NonHolonomicConfigurationPath:
        """Gets a non-linear (planned) configuration path given a target pose.

        :param position: The x, y, z position of the target.
        :param angle: The z orientation of the target (in radians).
        :param boundaries: A float defining the path search in x and y direction
        [[-boundaries,boundaries],[-boundaries,boundaries]].
        :param path_pts: number of sampled points returned from the computed path
        :param ignore_collisions: If collision checking should be disabled.
        :param algorithm: Algorithm used to compute path
        :raises: ConfigurationPathError if no path could be created.

        :return: A non-linear path (x,y,angle) in the xy configuration space.
        """

        path = self.get_nonlinear_path_points(
            position, angle, boundaries, path_pts, ignore_collisions, algorithm)

        return NonHolonomicConfigurationPath(self, path)

    def get_nonlinear_path_points(self, position: List[float],
                           angle=0,
                           boundaries=2,
                           path_pts=600,
                           ignore_collisions=False,
                           algorithm=Algos.RRTConnect) -> List[List[float]]:
        """Gets a non-linear (planned) configuration path given a target pose.

        :param position: The x, y, z position of the target.
        :param angle: The z orientation of the target (in radians).
        :param boundaries: A float defining the path search in x and y direction
        [[-boundaries,boundaries],[-boundaries,boundaries]].
        :param path_pts: number of sampled points returned from the computed path
        :param ignore_collisions: If collision checking should be disabled.
        :param algorithm: Algorithm used to compute path
        :raises: ConfigurationPathError if no path could be created.

        :return: A non-linear path (x,y,angle) in the xy configuration space.
        """

        # Base dummy required to be parent of the robot tree
        # self.base_ref.set_parent(None)
        # self.set_parent(self.base_ref)

        # Missing the dist1 for intermediate target

        self.target_base.set_position([position[0], position[1], self.target_z])
        self.target_base.set_orientation([0, 0, angle])

        handle_base = self.get_handle()
        handle_target_base = self.target_base.get_handle()

        # Despite verbosity being set to 0, OMPL spits out a lot of text
        with utils.suppress_std_out_and_err():
            _, ret_floats, _, _ = utils.script_call(
                'getNonlinearPathMobile@PyRep', PYREP_SCRIPT_TYPE,
                ints=[handle_base, handle_target_base,
                      self._collision_collection,
                      int(ignore_collisions), path_pts], floats=[boundaries],
                      strings=[algorithm.value])

        if len(ret_floats) == 0:
            raise ConfigurationPathError('Could not create path.')

        path = []
        for i in range(0, len(ret_floats) // 3):
            inst = ret_floats[3 * i:3 * i + 3]
            if i > 0:
                dist_change = sqrt((inst[0] - prev_inst[0]) ** 2 + (
                inst[1] - prev_inst[1]) ** 2)
            else:
                dist_change = 0
            inst.append(dist_change)

            path.append(inst)

            prev_inst = inst

        return path

    def step(self, path:NonHolonomicConfigurationPath) -> bool:
        """Make a step along the trajectory.

        Step forward by calling _get_base_actuation to get the velocity needed
        to be applied at the wheels.

        NOTE: This does not step the physics engine. This is left to the user.

        :return: If the end of the trajectory has been reached.

        """
        if self._path_done:
            raise RuntimeError('This path has already been completed. '
                               'If you want to re-run, then call set_to_start.')

        pos_inter = self.intermediate_target_base.get_position(
            relative_to=self)

        if len(path._path_points) > 2:  # Non-linear path
            if path.inter_done:
                path._next_i_path()
                path._set_inter_target(path.i_path)
                path.inter_done = False

            if sqrt((pos_inter[0]) ** 2 + (pos_inter[1]) ** 2) < 0.1:
                path.inter_done = True
                [vl, vr], _ = self.get_base_actuation()
            else:
                [vl, vr], _ = self.get_base_actuation()

            self.set_velocity(vl,vr,vl,vr)

            if path.i_path == len(path._path_points) - 1:
                self._path_done = True

        else:
            [vl, vr], self._path_done = self.get_base_actuation()
            self.set_velocity(vl,vr,vl,vr)

        return self._path_done

    def get_base_actuation(self):
        """A controller using PID.

        :return: A list with left and right joint velocity, and bool if target is reached.
        """

        d_x, d_y, _ = self.intermediate_target_base.get_position(
            relative_to=self)

        d_x_final, d_y_final, _ = self.target_base.get_position(
            relative_to=self)

        if sqrt((d_x_final) ** 2 + (d_y_final) ** 2) < 0.1:
            return [0., 0.], True

        alpha = atan2(d_y, d_x)
        e = atan2(sin(alpha), cos(alpha))
        e_P = e
        e_I = self.cummulative_error + e
        e_D = e - self.prev_error
        w = self.Kp * e_P + self.Ki * e_I + self.Kd * e_D
        w = atan2(sin(w), cos(w))

        self.cummulative_error = self.cummulative_error + e
        self.prev_error = e

        vr = ((2. * self.desired_velocity + w * self.wheel_distance) /
              (2. * self.wheel_radius))
        vl = ((2. * self.desired_velocity - w * self.wheel_distance) /
              (2. * self.wheel_radius))

        return [vl, vr], False

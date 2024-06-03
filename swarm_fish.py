import sys
sys.path.append('/home/gautier/dev/swarm/dronesim')
sys.path.append('/home/gautier/dev/swarm/UavSwarmFish')

from PyQt6.QtWidgets import QApplication
#from PyQt6.QtCore import Qt, QTimer

from swarm_controller import SwarmFish_Environment, SwamFish_View, SwarmFish_Controller, ms_of_hz, make_args_parser
import util

import numpy as np
import math

import swarmfish.swarm_control as sc
import swarmfish.obstacles as so

TEST_OBSTACLE = False

class SwarmFish_Scenario(SwarmFish_Controller):

    def __init__(self, ARGS, env, view, start=True):
        super().__init__(ARGS, env, view)

        #### Init SwarmFish ########################################
        arena_radius = 10.
        arena_center = np.array([0., 0., 0.])
        self.arena = so.Arena(center=arena_center[0:2], radius=arena_radius)
        self.view.add_cylinder(radius=arena_radius, height=0.1, pos=arena_center, color=(0,1,0,1))

        if TEST_OBSTACLE:
            obstacle_radius = 0.5
            obstacle_center = np.array([0, 4., 0])
            obstacle_z_min = 0.
            obstacle_z_max = 4.
            self.obstacle = so.CircleObstacle(obstacle_center[0:2], obstacle_radius, obstacle_z_min, obstacle_z_max)
            self.view.add_cylinder(radius=obstacle_radius,
                    height=obstacle_z_max-obstacle_z_min,
                    pos=obstacle_center)

        if start:
            self.start_simulation()


    def update_action(self):
        #### Step the simulation ###################################
        self.current_time += 1. / self.control_freq_hz

        states = { str(j): sc.State(
            self.obs[str(j)]["state"][0:3],
            self.obs[str(j)]["state"][10:13],
            self.obs[str(j)]["state"][9],
            self.current_time) for j in range(self.num_drones) }
        #### Compute control for the current state #############
        for uav_id in range(self.num_drones):
            # If you need : obs[str(j)]["state"] include jth vehicle's
            # position [0:3]  quaternion [3:7]   Attitude[7:10]  VelocityInertialFrame[10:13]     qpr[13:16]   motors[16:20]
            #   X  Y  Z       Q1   Q2   Q3  Q4   R  P   Y        VX     VY    VZ                  WX WY WZ     P0 P1 P2 P3
            uav_name = str(uav_id)
            state = states[uav_name]
            wall = self.arena.get_wall(state, self.params)
            if TEST_OBSTACLE:
                obstacle = self.obstacle.get_wall(state, self.params)
                if obstacle is not None and obstacle[0] < 2*self.params.l_w and obstacle[0] < wall[0]:
                    wall = obstacle
                    #print('Obstacle', uav_name, obstacle)
            if uav_id in self.intruders_id:
                cmd = sc.compute_interactions(state, self.params, [], nb_influent=0, wall=wall, altitude=5., z_min = 1., z_max = 10.)
            else:
                neighbors = [ states[str(k)] for k in range(self.num_drones) if (k != uav_id) and (k not in self.intruders_id)  ]
                intruders = [ states[str(k)] for k in self.intruders_id if k != uav_id ]
                cmd = sc.compute_interactions(state, self.params, neighbors, nb_influent=1, wall=wall,
                        altitude=self.altitude_setpoint, z_min = 1., z_max = 10.,
                        direction=self.direction_setpoint,
                        intruders=intruders)
            desired_course = sc.wrap_to_pi(state.get_course() + cmd.delta_course)
            #magnitude = min(max(self.speed_setpoint + cmd.delta_speed, self.params.min_velocity) / self.params.max_velocity, 1.)
            magnitude = self.speed_setpoint
            #magnitude = min(max(self.speed_setpoint + cmd.delta_speed, self.params.min_velocity) / self.params.max_velocity, 1.)
            if uav_id in self.intruders_id:
                magnitude *= 2.
            #speed = np.array([math.cos(desired_course), math.sin(desired_course), cmd.delta_vz, magnitude])
            speed = np.array([
                magnitude * math.cos(desired_course),
                magnitude * math.sin(desired_course),
                cmd.delta_vz,
                desired_course])
            self.commands[uav_id] = speed
            #self.action[uav_name] = speed # when actions are speeds directly

            #print('state',uav_id,state)
            #print(f' wall {uav_id} | dist {wall[0]:.2f}, angle= {np.degrees(wall[1]):.2f}')
            #print(f' cmd {uav_id} | {np.degrees(cmd[0]):0.2f}, {cmd[1]:0.2f}, {cmd[2]:0.2f} | desired_course {np.degrees(desired_course):0.2f}')
            #print(' speed',uav_id,speed)
        #print('') # blank line


if __name__ == "__main__":

    parser = make_args_parser()
    args = parser.parse_args()
    env = SwarmFish_Environment(args)
    app = QApplication(sys.argv)
    view = SwamFish_View()
    controller = SwarmFish_Scenario(args, env.env, view)
    app.exec()
    controller.close()
    view.close()
    env.close()
    sys.exit()

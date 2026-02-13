import sys

from PyQt6.QtWidgets import QApplication

from swarm_controller_simple_sim import SwarmFish_Environment, SwamFish_View, SwarmFish_Controller, ms_of_hz, make_args_parser

import numpy as np
import math
import time

import swarmfish.swarm_control as sc
import swarmfish.obstacles as so


TEST_OBSTACLE = False
SHOW_DIRECTION = True
SHOW_ARENA = True
SHOW_CELLS = True
SHOW_INFLUENTIALS = True
NB_INFLUENTIAL = 1

POS_NOISE = 0.
SPEED_NOISE = 0. #0.1
HEADING_NOISE = 0.
MAX_OVERFLY = 5
OVFY_PERIOD = 5. #s, minimum duration for overlfy to be registered

class Cell():

    def __init__(self, idx:int, idy:int, x:float, y:float, cell_lx:float, cell_ly:float):
        self.id = (idx, idy)
        self.overfly_time = time.time()
        self.overfly_count = 0
        self.position = (x, y)
        self.size = (cell_lx, cell_ly)
        self.vertices = np.array([[x, y], 
                                  [x+cell_lx, y], 
                                  [x, y+cell_ly], 
                                  [x+cell_lx, y+cell_ly]
                                  ])

    def __repr__(self) -> str:
        return f"Id: {self.id}, position: ({self.position}), size: ({self.size}), count: {self.overfly_count}"

    def overfly(self):
        time_since_last_ovfy = time.time() - self.overfly_time
        if time_since_last_ovfy > OVFY_PERIOD:
            if self.overfly_count < MAX_OVERFLY:
                self.overfly_count += 1
                self.overfly_time = time.time() # reset time since last overfly
        print(f"Cell n°: {self.id}, time since last overlfy: {time_since_last_ovfy}, overfly count: {self.overfly_count}") 
    # TODO: Add freshness to counter



class Exploration_Area_Rect():

    def __init__(self, lx:float, ly:float, nb_cells_x:int, nb_cells_y:int):
        self.total_size = lx * ly
        self.lx = lx
        self.ly = ly
        self.origin = np.array([0, 0])
        self.vertices = np.array([self.origin, 
                                  self.origin + np.array([lx,0.]), 
                                  self.origin + np.array([0., ly]), 
                                  self.origin + np.array([lx, ly])
                                  ])
        self.nb_cells = nb_cells_x * nb_cells_y
        self.nb_cells_x = nb_cells_x
        self.nb_cells_y = nb_cells_y
        self.cell_lx = lx/nb_cells_x
        self.cell_ly = ly/nb_cells_y
    
    def __repr__(self) -> str:
        return f"Size: {self.total_size}, origin: {self.origin}, #cells: {self.nb_cells}"

    def build_cells(self):
        '''
        Funtion populates self.cells with grid of cells.
        '''
        self.cells = []
        for i in range(self.nb_cells_x):
            temp_list = []
            for j in range(self.nb_cells_y):
                temp_list.append(Cell(i, j, self.origin[0]+i*self.cell_lx, self.origin[1]+j*self.cell_ly, self.cell_lx, self.cell_ly))
            self.cells.append(temp_list)
        self.cells = np.array(self.cells)
        

    def which_cell(self, x:float, y:float): 
        '''
        Funtion returns cell.id of the cell which has coordinates x and y.
        '''
        if self.origin[0] <= x <= self.origin[0]+self.lx:
            if self.origin[1] <= y <= self.origin[1]+self.ly:
                return (int(x//self.cell_lx), int(y//self.cell_ly))
        else:
            return None



class SwarmFish_Scenario(SwarmFish_Controller):

    def __init__(self, ARGS, env, view, start=True):
        super().__init__(ARGS, env, view)

        #### Init SwarmFish ########################################
        arena_radius = math.sqrt(2)*10.
        arena_center = np.array([0., 0., 0.])
        
        self.cell_arena = Exploration_Area_Rect(10., 10., 10, 10)
        self.cell_arena.build_cells()
        # TODO: Change drone arena from circle to square
        self.arena = so.Arena(center=arena_center[0:2], radius=arena_radius, name="arena")

        if SHOW_ARENA:
            self.view.add_polygon(vertices=self.cell_arena.vertices, height=0.01, color=(0,1,0,1))

        if SHOW_CELLS:
            for i in range(self.cell_arena.nb_cells_x):
                for j in range(self.cell_arena.nb_cells_y):
                    self.view.add_polygon(vertices=self.cell_arena.cells[i][j].vertices, height=min(0.1, 0.1+0.1*self.cell_arena.cells[i][j].overfly_count), color=(0,1,1,1))


        init_yaw = [ self.obs[j].att[2] for j in range(self.num_drones) ]
        #self.desired_course = np.array(init_yaw) # np.zeros(self.num_drones)

        if TEST_OBSTACLE:
            obstacle_radius = 0.5
            obstacle_center = np.array([0, 4., 0])
            obstacle_z_min = 0.
            obstacle_z_max = 4.
            self.obstacle = so.CircleObstacle(obstacle_center[0:2], obstacle_radius, obstacle_z_min, obstacle_z_max, name='pole')
            self.view.add_cylinder(radius=obstacle_radius,
                    height=obstacle_z_max-obstacle_z_min,
                    pos=obstacle_center)
            polygon_vertices = np.array([[1., -1.], [1., 1.], [-2., 1.], [-1., -1.]]) + np.full((4,2),np.array([-3., -4.]))
            self.polygon = so.PolygonObstacle(polygon_vertices, obstacle_z_min, obstacle_z_max, name="polygon")
            self.view.add_polygon(vertices=polygon_vertices, height=obstacle_z_max-obstacle_z_min)

        if SHOW_INFLUENTIALS:
            self.lines = {}
            for d in range(env.num_drones):
                self.lines[d] = [ self.view.add_line(np.zeros(3), np.zeros(3)) for x in range(NB_INFLUENTIAL) ]

        if SHOW_DIRECTION:
            self.directions = {}
            self.speeds = {}
            for d in range(env.num_drones):
                self.directions[d] = [ self.view.add_line(np.zeros(3), np.zeros(3), color=(1.,0.,0.,1.)) ]
                self.speeds[d] = [ self.view.add_line(np.zeros(3), np.zeros(3), color=(0.,0.,1.,1.)) ]

        if start:
            self.start_simulation()


    def update_action(self):
        #### Step the simulation ###################################

        noise = np.random.normal(size=7)
        states = { j: sc.State(
            self.obs[j].pos + POS_NOISE*noise[0:3],
            self.obs[j].vel + SPEED_NOISE*noise[3:6],
            self.obs[j].att[2] + HEADING_NOISE*noise[6],
            self.current_time, str(j)) for j in range(self.num_drones) }
        #### Compute control for the current state #############
        for uav_id in range(self.num_drones):
            # If you need : obs[str(j)]["state"] include jth vehicle's
            # position [0:3]  quaternion [3:7]   Attitude[7:10]  VelocityInertialFrame[10:13]     qpr[13:16]   motors[16:20]
            #   X  Y  Z       Q1   Q2   Q3  Q4   R  P   Y        VX     VY    VZ                  WX WY WZ     P0 P1 P2 P3
            uav_name = str(uav_id)
            state = states[uav_id]
            wall = self.arena.get_wall(state, self.params)
            obstacles = []
            if TEST_OBSTACLE:
                obstacles = [ o.get_wall(state, self.params) for o in [self.obstacle, self.polygon] if o is not None ]
            if uav_id in self.intruders_id:
                cmd, _ = sc.compute_interactions(state, self.params, [], nb_influent=0,
                        wall=wall, altitude=5., z_min=1., z_max=10., obstacles=obstacles)
            else:
                neighbors = [ states[k] for k in range(self.num_drones) if (k != uav_id) and (k not in self.intruders_id)  ]
                intruders = [ states[k] for k in self.intruders_id if k != uav_id ]
                cmd, influentials = sc.compute_interactions(state, self.params, neighbors, nb_influent=NB_INFLUENTIAL, wall=wall,
                        altitude=self.altitude_setpoint, z_min = 1., z_max = 10.,
                        direction=self.direction_setpoint,
                        intruders=intruders, obstacles=obstacles)
                if SHOW_INFLUENTIALS:
                    for l, influential in zip(self.lines[uav_id], influentials):
                        self.view.move_line(l, state.pos, states[int(influential[1])].pos)
            #desired_course = sc.wrap_to_pi(state.get_course() + cmd.delta_course)
            #self.desired_course[uav_id] = sc.wrap_to_pi(self.desired_course[uav_id] + cmd.delta_course / self.simulation_freq_hz)
            #desired_course = self.desired_course[uav_id] + 0.1*np.random.rand()
            yaw_rate = cmd.delta_course #/ self.simulation_freq_hz
            #print(uav_id, yaw_rate)
            #print(f'desired course {uav_name}: {np.degrees(desired_course):0.2f} | {np.degrees(state.get_course()):0.2f} + {np.degrees(cmd.delta_course):0.2f}')
            # TODO fix heading/rates
            magnitude = self.speed_setpoint + cmd.delta_speed # TODO clip min/max
            if uav_id in self.intruders_id:
                magnitude *= 2.
            speed = np.array([
                magnitude * math.cos(state.get_course(use_heading=True)),
                magnitude * math.sin(state.get_course(use_heading=True)),
                cmd.delta_vz,
                yaw_rate])
            self.commands[uav_id] = speed
            
            # Compute & draw overflown cell
            cell_id = controller.cell_arena.which_cell(state.pos[0], state.pos[1])
            if cell_id is not None:
                controller.cell_arena.cells[cell_id[0]][cell_id[1]].overfly()


            if SHOW_DIRECTION:
                for d, s in zip(self.directions[uav_id], self.speeds[uav_id]):
                    self.view.move_line(d, state.pos, state.pos+speed[0:3])
                    self.view.move_line(s, state.pos, state.pos+state.speed)

            #print(state)
            #print(f' wall {uav_id} | dist {wall[0]:.2f}, angle= {np.degrees(wall[1]):.2f}')
            #print(f' cmd {uav_id} | {np.degrees(cmd.delta_course):0.2f}, {cmd.delta_speed:0.3f}, {cmd.delta_vz:0.3f} | desired_course {np.degrees(desired_course):0.2f}')
            #print(' speed',uav_id,speed)
        #print('') # blank line


if __name__ == "__main__":

    parser = make_args_parser()
    args = parser.parse_args()
    env = SwarmFish_Environment(args)
    app = QApplication(sys.argv)
    view = SwamFish_View()
    controller = SwarmFish_Scenario(args, env, view)
    app.exec()
    controller.close()
    view.close()
    env.close()
    sys.exit()

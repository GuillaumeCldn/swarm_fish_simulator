import sys

from PyQt6.QtWidgets import QApplication

from swarm_controller_simple_sim import SwarmFish_Environment, SwamFish_View, SwarmFish_Controller, make_args_parser

import numpy as np
import math
import time

import swarmfish.swarm_control as sc
import swarmfish.obstacles as so


TEST_OBSTACLE = False
SHOW_DIRECTION = True
SHOW_EE_AREA = True
SHOW_CELLS = True
SHOW_FOV = True
SHOW_INFLUENTIALS = True
NB_INFLUENTIAL = 1

POS_NOISE = 0.
SPEED_NOISE = 0. # 0.1
HEADING_NOISE = 0.

OVFY_PERIOD = 10. # s, minimum duration for overfly to be registered
SPOIL_TIME = 60. # s, time after which spoilage should start increasing rapidly
MAX_SPOIL = 100. # maximum spoilage value
FRESHEN_RATE = MAX_SPOIL/4. # amount by which spoilage is decreased when a cell is overflown

CELL_HMIN = 0.1
CELL_HMAX = 1.
ALPHA = CELL_HMAX/MAX_SPOIL # rate at which the cell height is updated

SENSOR_VIEW_HEIGHT = 10. # m, height at which sensor resolution is average
SENSOR_VIEW_ANGLE = 60. # °, aperture of sensor view cone

FOV_COLOUR = (0.,0.,1,0.1)
EE_AREA_COLOUR = (0.,1.,0.,1.)
ARENA_COLOUR = (1.,0.,0.,1.)
CELL_COLOUR = (0.,1.,1.,1.)

ARENA_RADIUS = 10. # m
EE_AREA_LX = EE_AREA_LY = 20. # m
NB_CELLS_X = NB_CELLS_Y = 20


class Cell():

    def __init__(self, idx:int, idy:int, x:float, y:float, cell_lx:float, cell_ly:float, init_time:float):
        self.id = (idx, idy)
        self.last_ovfy_time = init_time
        self.spoilage = MAX_SPOIL
        self.position = (x, y)
        self.size = (cell_lx, cell_ly)
        self.vertices = np.array([[x, y], 
                                  [x+cell_lx, y], 
                                  [x+cell_lx, y+cell_ly],
                                  [x, y+cell_ly] 
                                  ])
        self.height = CELL_HMIN
        self.mesh = None

    def __repr__(self) -> str:
        return f"Id: {self.id}, position: ({self.position}), size: ({self.size}), spoilage: {self.spoilage}"

    def calc_height(self) -> None:
        '''
        Method updates the height of the cell proportionally to the spoilage.
        Cell height is constrained by CELL_HMIN and CELL_HMAX. 
        '''
        self.height = min(CELL_HMIN, CELL_HMAX - ALPHA*self.spoilage)

    def overfly(self, drone_height:float) -> None:
        '''
        Method updates the time of last overfly of the cells and calls the freshen() method. 
        '''
        time_since_last_ovfy = time.time() - self.last_ovfy_time
        if time_since_last_ovfy > OVFY_PERIOD:
                self.last_ovfy_time = time.time() # reset time since last overfly
                self.freshen(drone_height)

    # WARN: Spoilage rate climbs to fast
    # TODO: Find better function to update spoilage rate
    def spoil(self) -> None:
        '''
        Method increases cell spoilage exponentially over time until spoilage reaches MAX_SPOIL.
        '''
        spoil_increase = math.exp(time.time()-self.last_ovfy_time-SPOIL_TIME)
        if self.spoilage + spoil_increase < MAX_SPOIL:
            self.spoilage += spoil_increase
        else:
            self.spoilage = MAX_SPOIL
        self.calc_height()

    def freshen(self, drone_height:float) -> None:
        '''
        Method decreases spoilage, thus "freshening" the cell.
        To model data precision, the amount by which spoilage is decreased depends on the drone height.
        SENSOR_VIEW_HEIGHT models the height at which the sensor's resolution is "average". It is used to 
        calibrate the amount by which the cell is "freshened".
        '''
        freshen_amount = FRESHEN_RATE*math.exp(-drone_height/(SENSOR_VIEW_HEIGHT*2))
        if self.spoilage - freshen_amount > 0:
            self.spoilage -= freshen_amount
        else:
            self.spoilage = 0
        self.calc_height()


class Exploration_Area_Rect():

    def __init__(self, lx:float, ly:float, nb_cells_x:int, nb_cells_y:int):
        self.total_size = lx * ly
        self.lx = lx
        self.ly = ly
        self.origin = np.array([0., 0.])
        self.vertices = np.array([self.origin, 
                                  self.origin + np.array([lx,0.]), 
                                  self.origin + np.array([lx, ly]),
                                  self.origin + np.array([0., ly]) 
                                  ])
        self.nb_cells = nb_cells_x * nb_cells_y
        self.nb_cells_x = nb_cells_x
        self.nb_cells_y = nb_cells_y
        self.cell_lx = lx/nb_cells_x
        self.cell_ly = ly/nb_cells_y
        self.init_time = time.time()
        self.min_spoilage = float("inf")
    
    def __repr__(self) -> str:
        return f"Size: {self.total_size}, origin: {self.origin}, #cells: {self.nb_cells}"

    def build_cells(self) -> None:
        '''
        Method populates self.cells with grid of cells.
        '''
        self.cells = []
        for i in range(self.nb_cells_x):
            temp_list = []
            for j in range(self.nb_cells_y):
                temp_list.append(Cell(i, j, self.origin[0]+i*self.cell_lx, self.origin[1]+j*self.cell_ly, self.cell_lx, self.cell_ly, self.init_time-OVFY_PERIOD))
            self.cells.append(temp_list)
        self.cells = np.array(self.cells)

    def in_EE(self, x:float, y:float) -> bool:
        '''
        Method checks if coordinates are in the exploration area. 
        '''
        if (self.origin[0] <= x <= self.origin[0]+self.lx) and (self.origin[1] <= y <= self.origin[1]+self.ly):
            return True
        else:
            return False
    
    def id_in_EE(self, idx:int, idy:int) -> bool:
        '''
        Method checks if a cell id is in the exploration area. 
        '''
        if (0<=idx<self.nb_cells_x) and (0<=idy<self.nb_cells_y):
            return True
        else:
            return False

    def coords_to_id(self, x:float, y:float) -> tuple[int, int]:
        '''
        Method returns the id of the cell which contains coordinates x and y.
        '''
        return (int(x//self.cell_lx), int(y//self.cell_ly))

    def which_cells(self, x:float, y:float, z: float) -> list[tuple[int, int]] | None: 
        '''
        Method returns list of ids of cells which are in the drone's sensor fov.
        '''
        cells = []
        view_radius = z*math.tan(np.radians(SENSOR_VIEW_ANGLE/2.))
        cell_count_x = int(view_radius // self.cell_lx)
        cell_count_y = int(view_radius // self.cell_ly)
        # Check if drone is above the exploration area
        if self.in_EE(x, y):
                idx_0, idy_0 = self.coords_to_id(x, y)
                # Add all cells except corner cells
                for i in range(-cell_count_x, cell_count_x+1):
                    for j in range(-cell_count_y, cell_count_y+1):
                        if (np.abs(i) == cell_count_x) and (np.abs(j) == cell_count_y):
                            pass
                        else:
                            if self.id_in_EE(idx_0+i, idy_0+j):
                                cells.append((idx_0+i, idy_0+j))
                # Test if corner cells are in fov
                corner_coord = math.sqrt(2)/2.*view_radius
                for k in range(-1, 2, 2):
                    for l in range(-1, 2, 2):
                        if self.in_EE(x+k*corner_coord, y+l*corner_coord): 
                            corner_candidate = self.coords_to_id(x+k*corner_coord, y+l*corner_coord)
                            if corner_candidate not in cells:
                                cells.append(corner_candidate)
                            else:
                                pass
        else:
            return None
        return cells

    def spoil_cells(self) -> None:
        '''
        Method spoils each cell in the arena.
        '''
        for i in range(self.nb_cells_x):
            for j in range(self.nb_cells_y):
                self.cells[i][j].spoil()


class SwarmFish_Scenario(SwarmFish_Controller):

    def __init__(self, ARGS, env, view, start=True):
        super().__init__(ARGS, env, view)

        #### Init SwarmFish ########################################
        # WARN: If arena is centred, cell behaviour is wrong
        arena_radius = math.sqrt(2)*ARENA_RADIUS
        arena_center = np.array([ARENA_RADIUS, ARENA_RADIUS, 0.])
        
        self.cell_arena = Exploration_Area_Rect(lx=EE_AREA_LX, ly=EE_AREA_LY, nb_cells_x=NB_CELLS_X, nb_cells_y=NB_CELLS_Y)
        self.cell_arena.build_cells()
        # TODO: Change drone arena from circle to square
        self.arena = so.Arena(center=arena_center[0:2], radius=arena_radius, name="arena")
        self.view.add_cylinder(radius=arena_radius, height=0.01, pos=arena_center, color=ARENA_COLOUR)
        if SHOW_EE_AREA:
            self.view.add_polygon(vertices=self.cell_arena.vertices, height=0.05, color=EE_AREA_COLOUR)
        if SHOW_CELLS:
            self.draw_cells(init=True)

        if SHOW_FOV:
            self.fov_dict = {}
            for d in range(env.num_drones):
                self.draw_fov(uav_id=d, pos=np.array([0., 0., 0.]), init=True)
                
        #init_yaw = [ self.obs[j].att[2] for j in range(self.num_drones) ]
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
                self.lines[d] = [ self.view.add_line(np.zeros(3), np.zeros(3)) for _ in range(NB_INFLUENTIAL) ]

        if SHOW_DIRECTION:
            self.directions = {}
            self.speeds = {}
            for d in range(env.num_drones):
                self.directions[d] = [ self.view.add_line(np.zeros(3), np.zeros(3), color=(1.,0.,0.,1.)) ]
                self.speeds[d] = [ self.view.add_line(np.zeros(3), np.zeros(3), color=(0.,0.,1.,1.)) ]

        if start:
            self.start_simulation()

    def draw_cells(self, init:bool=False) -> None:
        '''
        Method draws each cell in the arena.
        By default, the method updates already drawn cells.
        If init is set to True, the cell is drawn for the first time.
        '''
        for i in range(self.cell_arena.nb_cells_x):
            for j in range(self.cell_arena.nb_cells_y):
                cell = self.cell_arena.cells[i][j]
                if init:
                    cell.mesh = self.view.build_polygon_mesh(vertices=cell.vertices, height=cell.height, color=CELL_COLOUR) 
                    self.view.add_mesh(cell.mesh)
                else:
                    old_mesh = cell.mesh
                    cell.mesh = self.view.build_polygon_mesh(vertices=cell.vertices, height=cell.height, color=CELL_COLOUR) 
                    self.view.update_mesh(old_mesh, cell.mesh)

    def draw_fov(self, uav_id:int, pos:np.ndarray, init:bool=False) -> None:
        view_radius = pos[2]*math.tan(np.radians(SENSOR_VIEW_ANGLE/2.))
        if init:
            self.fov_dict[uav_id] = self.view.build_cone_mesh(view_radius, pos, color=FOV_COLOUR)
            self.view.add_mesh(self.fov_dict[uav_id])
        else:
            old_fov = self.fov_dict[uav_id]
            self.fov_dict[uav_id] = self.view.build_cone_mesh(view_radius, pos, color=FOV_COLOUR)
            self.view.update_mesh(old_fov, self.fov_dict[uav_id])

    def measure_spoilage(self) -> float:
        '''
        Method calculates the normalised spoilage of all cells in the exploration arena.
        '''
        spoilage = 0
        total_spoilage = MAX_SPOIL*self.cell_arena.nb_cells
        for i in range(self.cell_arena.nb_cells_x):
            for j in range(self.cell_arena.nb_cells_y):
                cell = self.cell_arena.cells[i][j]
                spoilage += cell.spoilage
        current_spoilage = spoilage/total_spoilage
        if self.cell_arena.min_spoilage > current_spoilage:
            self.cell_arena.min_spoilage = current_spoilage
        return current_spoilage

    def update_action(self):
        #### Step the simulation ###################################
        self.draw_cells()
        noise = np.random.normal(size=7)
        states = { j: sc.State(
            self.obs[j].pos + POS_NOISE*noise[0:3],
            self.obs[j].vel + SPEED_NOISE*noise[3:6],
            self.obs[j].att[2] + HEADING_NOISE*noise[6],
            self.current_time, str(j)) for j in range(self.num_drones) }
        self.cell_arena.spoil_cells()
        #### Compute control for the current state #############
        for uav_id in range(self.num_drones):
            '''
            If you need : obs[str(j)]["state"] include jth vehicle's
            position [0:3]  quaternion [3:7]   Attitude[7:10]  VelocityInertialFrame[10:13]     qpr[13:16]   motors[16:20]
            X  Y  Z         Q1   Q2   Q3  Q4   R  P  Y         VX     VY     VZ                 WX WY WZ     P0 P1 P2 P3
            '''
            #uav_name = str(uav_id)
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
            
            # Compute overflown cell
            cells = self.cell_arena.which_cells(state.pos[0], state.pos[1], state.pos[2])
            if cells is not None:
                for cell_id in cells:
                    if cell_id is not None:
                        # WARN: Some cells seem to be drawn as overflown late
                        self.cell_arena.cells[cell_id[0]][cell_id[1]].overfly(state.pos[2])


            if SHOW_DIRECTION:
                for d, s in zip(self.directions[uav_id], self.speeds[uav_id]):
                    self.view.move_line(d, state.pos, state.pos+speed[0:3])
                    self.view.move_line(s, state.pos, state.pos+state.speed)

            if SHOW_FOV:
                self.draw_fov(uav_id, state.pos)

            print(f"Minimum spoilage = {self.cell_arena.min_spoilage*100:.2f}, current spoilage = {self.measure_spoilage()*100:.2f}%")


if __name__ == "__main__":

    # TODO: Display spoilage measurement.
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

import sys
sys.path.append('/home/gautier/dev/swarm/dronesim')
sys.path.append('/home/gautier/dev/swarm/UavSwarmFish')

import pybullet as p
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QDockWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QSlider
from PyQt6.QtCore import Qt, QTimer
from ui_swarm_controller import Ui_SwarmController

import pyqtgraph as pg
import pyqtgraph.opengl as gl
import util

import argparse
import dataclasses
import numpy as np
import math

from dronesim.control.INDIControl import INDIControl
from dronesim.envs.BaseAviary import DroneModel, Physics
from dronesim.envs.VelocityAviary import VelocityAviary
from dronesim.envs.CtrlAviary import CtrlAviary
from dronesim.utils.Logger import Logger
from dronesim.utils.trajGen import *
from dronesim.utils.utils import str2bool, sync

from swarmfish.utils import load_params_from_yaml
import swarmfish.swarm_control as sc

def ms_of_hz(freq):
    return int(1000 / freq)

SIMULATION_FREQ = 100 # in Hz
CONTROL_FREQ = 100  # in HZ
NB_OF_DRONES = 5


class SwarmFish_Environment():
    env = None

    def __init__(self, ARGS, create_env=True):
        self.num_drones = ARGS.num_drones
        self.drone = ARGS.drone
        self.simulation_freq_hz = ARGS.simulation_freq_hz
        self.control_freq_hz = ARGS.control_freq_hz
        self.aggregate = ARGS.aggregate
        self.gui = ARGS.gui
        self.record_video = ARGS.record_video
        self.obstacles = ARGS.obstacles
        self.user_debug_gui = ARGS.user_debug_gui

        if create_env:
            self.create()

    def create(self):

        #### Initialize the simulation #############################
        H = 0.50
        H_STEP = 0.05
        R = 1.5

        AGGR_PHY_STEPS = (
            int(self.simulation_freq_hz / self.control_freq_hz) if self.aggregate else 1
        )
        INIT_XYZS = np.array(
            [
                [
                    R * np.cos((i / 6) * 2 * np.pi + np.pi / 2),
                    R * np.sin((i / 6) * 2 * np.pi + np.pi / 2),
                    H + i * H_STEP,
                ]
                for i in range(self.num_drones)
            ]
        )
        INIT_RPYS = np.array([[0.0, 0.0, 0.0] for i in range(self.num_drones)])

        #### Create the environment
        self.env = CtrlAviary(
            drone_model=self.num_drones * self.drone,
            num_drones=self.num_drones,
            initial_xyzs=INIT_XYZS,
            initial_rpys=INIT_RPYS,
            physics=Physics.PYB,
            neighbourhood_radius=10,
            freq=self.simulation_freq_hz,
            aggregate_phy_steps=AGGR_PHY_STEPS,
            gui=self.gui,
            record=self.record_video,
            obstacles=self.obstacles,
            user_debug_gui=self.user_debug_gui,
        )

    def close(self):
        print("Closing env")
        if self.env is not None:
            self.env.close()


class SwamFish_View(QMainWindow):
    mesh_list = {}

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Swarm Fish View")
        self.resize(1024,768)
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor(100,100,100, 255)
        grid = gl.GLGridItem()
        grid.setColor('k')
        self.view.addItem(grid)
        self.setCentralWidget(self.view)
        self.show()

    def close(self):
        print("Closing view")
        QApplication.exit()

    # Create a cylinder in pybullet format
    def create_cylinder(self, radius:float, height:float, pos:np.ndarray, res:int=20):
        t_ = np.linspace(0, 2*np.pi, res)
        x_ = pos[0] + radius * np.sin(t_)
        y_ = pos[1] + radius * np.cos(t_)
        bottom = np.array([[x,y,pos[2]] for x,y in zip(x_,y_)])
        top = np.array([[x,y,pos[2]+height] for x,y in zip(x_,y_)])
        return np.vstack((bottom,top))
    
    # Add polygons in pybullet
    def add_polygon_to_env(self, vertices):
        polygon_id = p.createCollisionShape(p.GEOM_MESH,vertices=vertices)
        p.createMultiBody(baseCollisionShapeIndex=polygon_id)

    def add_cylinder(self, radius:float, height:float, pos:np.ndarray, res:int=20, color=(1.,1.,1.,1.)):
        verts, faces = util.cylinder_mesh(radius, height, res)
        mesh = gl.GLMeshItem(vertexes=verts,faces=faces,drawFaces=True,
                    drawEdges=False,smooth=False,computeNormals=True,
                    shader='shaded',glOptions='opaque')
        mesh.setColor(color)
        mesh.translate(pos[0], pos[1], pos[2])
        self.view.addItem(mesh)
        # also add to pybullet
        self.add_polygon_to_env(self.create_cylinder(radius, height, pos, res))

    def add_polygon(self, vertices:np.ndarray, height:float, color=(1., 1., 1., 1.)):
        verts, faces = util.polygon_mesh(vertices, height)
        mesh = gl.GLMeshItem(vertexes=verts,faces=faces,drawFaces=True,
                    drawEdges=False,smooth=False,computeNormals=True,
                    shader='shaded',glOptions='opaque')
        mesh.setColor(color)
        self.view.addItem(mesh)
        # also add to pybullet
        self.add_polygon_to_env(verts)


    def add_object(self, mesh_id, mesh, text=None):
        v,f = util.box_mesh((0.1,0.1,0.1))
        m0 =  gl.GLMeshItem(vertexes=v,faces=f,drawFaces=False,
                    drawEdges=False,smooth=False,computeNormals=False,shader=None)
        self.view.addItem(m0)
        for i, m in enumerate(mesh):
            m.setParentItem(m0)
            self.view.addItem(m)
        if text is not None:
            t = gl.GLTextItem(parentItem=m0, pos=[0,0,0.2], text=text)
            self.view.addItem(t)
        self.mesh_list[mesh_id] = [m0] + mesh
        self.move_object(mesh_id)
        self.view.update()

    def move_object(self, obj_id):
        pos, quat = p.getBasePositionAndOrientation(int(obj_id))
        angle,x,y,z = util.quaternion2axis_angle(quat)
        roll,pitch,yaw = util.quaternion2roll_pitch_yaw(quat)
        self.mesh_list[obj_id][0].resetTransform()
        self.mesh_list[obj_id][0].rotate(np.degrees(angle),x,y,z,local=True)
        self.mesh_list[obj_id][0].translate(pos[0],pos[1],pos[2])

    def add_line(self, pos1, pos2, color=(255, 255, 255, 255)):
        line = gl.GLLinePlotItem(pos=np.vstack((pos1, pos2)), color=color)
        self.view.addItem(line)
        return line

    def move_line(self, line, pos1, pos2, color=(255, 255, 255, 255)):
        line.setData(pos=np.vstack((pos1, pos2)), color=color)


class SwarmFish_Controller(QWidget, Ui_SwarmController):
    speed_setpoint = 1.
    direction_setpoint = None
    altitude_setpoint = 5.
    intruders_id = []

    def __init__(self, ARGS, env, view):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("SwarmFish")

        self.num_drones = ARGS.num_drones
        self.simulation_freq_hz = ARGS.simulation_freq_hz
        self.control_freq_hz = ARGS.control_freq_hz
        self.drone = ARGS.drone
        self.env = env
        self.view = view

        self.simulation_timer = QTimer()
        self.simulation_timer.timeout.connect(self.update_simulation)
        self.action_timer = QTimer()
        self.action_timer.timeout.connect(self.update_action)

        #### Initialize the controllers ############################
        self.ctrl = [INDIControl(drone_model=drone) for drone in self.env.DRONE_MODEL]
        self.commands = [ np.zeros(4) for i in range(self.num_drones) ]
        self.action = {str(i): np.zeros(4) for i in range(self.num_drones)}
        self.current_time = 0.
        self.obs, _, _, _ = self.env.step(self.action)

        for d in env.DRONE_IDS:
            mesh = util.bullet2pyqtgraph(d)
            view.add_object(d, mesh, text=str(d-1)) # FIXME shift due to ground plane in pybullet

        #### load params from config file 
        self.params = load_params_from_yaml(ARGS.swarm_config)

        self.make_layout()
        self.show()

    def update_simulation(self):
        self.current_time += 1. / self.simulation_freq_hz
        for k, v in enumerate(self.commands):
            state = self.obs[str(k)]["state"]
            cmd, _, _ = self.ctrl[k].computeControl(
                control_timestep=self.env.AGGR_PHY_STEPS * self.env.TIMESTEP,
                cur_pos=state[0:3],
                cur_quat=state[3:7],
                cur_vel=state[10:13],
                cur_ang_vel=state[13:16],
                target_pos=state[0:3], # same as the current position
                target_rpy=np.array([0., 0., v[3]]),  # keep current yaw
                target_vel=v[0:3] # target the desired velocity vector
            )
            self.action[str(k)] = cmd
        self.obs, _, _, _ = self.env.step(self.action)
        for d in self.env.DRONE_IDS:
            self.view.move_object(d)

    def update_action(self):
        raise NotImplementedError

    def start_simulation(self):
        self.action_timer.start(ms_of_hz(self.control_freq_hz))
        self.simulation_timer.start(ms_of_hz(self.simulation_freq_hz))
        print("simultion started")

    def stop_simulation(self):
        self.simulation_timer.stop()
        self.action_timer.stop()
        print("simultion stoped")

    def close(self):
        print("Closing controller")
        self.stop_simulation()

    def make_layout(self):
        CURSOR_RES = 1000

        # Boutons
        self.btn_start.clicked.connect(lambda: self.start_simulation())
        self.btn_stop.clicked.connect(lambda: self.stop_simulation())
        self.btn_quit.clicked.connect(lambda: self.view.close())

        # Speed
        def update_speed(_v):
            self.speed_setpoint = _v / 100.
            self.label_speed.setText(str(_v / 100.))
        self.slider_speed.valueChanged.connect(update_speed)

        # Direction
        def update_direction(_v):
            if self.check_dir.isChecked():
                self.direction_setpoint = np.radians(_v)
                self.label_dir.setText(str(_v))
            else:
                self.direction_setpoint = None
        self.slider_dir.valueChanged.connect(update_direction)
        self.check_dir.stateChanged.connect(lambda _: update_direction(self.slider_dir.value()))

        # Altitude
        def update_alt(_v):
            if self.check_alt.isChecked():
                self.altitude_setpoint = _v / 10.
                self.label_alt.setText(str(_v / 10.))
            else:
                self.altitude_setpoint = None
        self.slider_alt.valueChanged.connect(update_alt)
        self.check_alt.stateChanged.connect(lambda _: update_alt(self.slider_alt.value()))

        # Intruders
        def update_intruders():
            try:
                self.intruders_id = [int(i) for i in self.intruders_input.text().split(',')]
                print(f'Set intruders ID: {self.intruders_id}')
            except:
                self.intruders_id = []
                print('Invalid or no intruders ID')
        self.intruders_input.returnPressed.connect(update_intruders)

        try:
            # Parameters
            p_dict = dataclasses.asdict(self.params)
            exclude = ['max_velocity','min_velocity','zmax','zmin','use_heading']
            parameters = [ p for p in p_dict.keys() if p not in exclude ]

            container_widget = QWidget()
            param_layout = QVBoxLayout()

            for label_text in parameters:
                label = QLabel(label_text)
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setMinimum(0)
                slider.setMaximum(int(max(p_dict[label_text] * 5 * CURSOR_RES,5)))
                slider.setValue(int(p_dict[label_text] * CURSOR_RES))

                # Label pour afficher la valeur courante du slider
                slider_label = QLabel(str(p_dict[label_text]))
                horiz = QHBoxLayout()
                horiz.addWidget(label)
                horiz.addWidget(slider)
                horiz.addWidget(slider_label)

                def dataclass_from_dict(da, d):
                    try:
                        fieldtypes = {f.name:f.type for f in dataclasses.fields(da)}
                        return da(**{f:dataclass_from_dict(fieldtypes[f],d[f]) for f in d})
                    except:
                        return d # Not a dataclass field

                # Mettre à jour le label lorsque la valeur du slider change
                def update_val(value,l=label_text,s=slider_label):
                    var = float(value / CURSOR_RES)
                    tmp = dataclasses.asdict(self.params)
                    tmp[l] = var
                    self.params = dataclass_from_dict(sc.SwarmParams, tmp)
                    s.setText(str(value / CURSOR_RES))
                slider.valueChanged.connect(update_val)
                param_layout.addLayout(horiz)

            container_widget.setLayout(param_layout)
            self.params_area.setWidget(container_widget)
        except Exception as e:
            print("Fail to load parameters",e)
            sys.exit(0)

        self.show()


def make_args_parser():
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(
        description="SwarmFish control with Qt"
    )
    parser.add_argument(
        "--drone",
        default=["robobee"],
        type=list,
        help="Drone model",
        metavar="",
        choices=[DroneModel],
    )
    parser.add_argument(
        "--num_drones",
        default=NB_OF_DRONES,
        type=int,
        help="Number of drones",
        metavar="",
    )
    parser.add_argument(
        "--gui",
        default=False,
        type=str2bool,
        help="Whether to use PyBullet GUI",
        metavar="",
    )
    parser.add_argument(
        "--record_video",
        default=False,
        type=str2bool,
        help="Whether to record a video",
        metavar="",
    )
    parser.add_argument(
        "--user_debug_gui",
        default=False,
        type=str2bool,
        help="Whether to add debug lines and parameters to the GUI",
        metavar="",
    )
    parser.add_argument(
        "--aggregate",
        default=True,
        type=str2bool,
        help="Whether to aggregate physics steps",
        metavar="",
    )
    parser.add_argument(
        "--obstacles",
        default=False,
        type=str2bool,
        help="Whether to add obstacles to the environment",
        metavar="",
    )
    parser.add_argument(
        "--simulation_freq_hz",
        default=SIMULATION_FREQ,
        type=int,
        help="Simulation frequency in Hz",
        metavar="",
    )
    parser.add_argument(
        "--control_freq_hz",
        default=CONTROL_FREQ,
        type=int,
        help="Control frequency in Hz",
        metavar="",
    )
    parser.add_argument(
        "--swarm_config",
        default='config/demo.yaml',
        type=str,
        help="SwarmFish parameter file",
        metavar="",
    )
    return parser


if __name__ == "__main__":
    pass


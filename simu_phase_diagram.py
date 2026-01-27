"""Script for testing swarmfish with quadrotors following a desired velocity vector (unit vector + magnitude normalized by max vehicle speed in Km/h)
"""
import sys
import os
import argparse
import time
from datetime import datetime
import numpy as np
import math

from swarm_controller import SwarmFish_Environment, ms_of_hz, make_args_parser
import util

from dronesim.control.INDIControl import INDIControl
from dronesim.envs.BaseAviary import DroneModel, Physics
from dronesim.envs.CtrlAviary import CtrlAviary
from dronesim.utils.Logger import Logger
from dronesim.utils.utils import str2bool, sync

from swarmfish.utils import load_params_from_yaml, compute_quantification
import swarmfish.swarm_control as sc


def run_simulation(ARGS: dict):
    #### Initialize the simulation #############################
    dim = math.ceil(math.sqrt(ARGS.num_drones))
    dist = 2.

    AGGR_PHY_STEPS = (
        int(ARGS.simulation_freq_hz / ARGS.control_freq_hz) if ARGS.aggregate else 1
    )
    INIT_XYZS = np.array([[ dist*(i % dim), dist*math.floor(i / dim), 0. ] for i in range(ARGS.num_drones)]) 
    if ARGS.random_init:
        INIT_YAW = 2.*np.pi*np.random.rand(ARGS.num_drones)
    else:
        INIT_YAW = np.zeros(ARGS.num_drones)
    INIT_RPYS = np.array([[0.0, 0.0, INIT_YAW[i]] for i in range(ARGS.num_drones)])

    NOISES = [ np.radians(0.2), 0.01, 0.05 ] # noise on heading, speed and vertical speed (standard deviation of a normal distribution)

    #### Create the environment
    env = CtrlAviary(
        drone_model=ARGS.num_drones * ARGS.drone,
        num_drones=ARGS.num_drones,
        initial_xyzs=INIT_XYZS,
        initial_rpys=INIT_RPYS,
        physics=Physics.PYB,
        neighbourhood_radius=10,
        freq=ARGS.simulation_freq_hz,
        aggregate_phy_steps=AGGR_PHY_STEPS,
        gui=ARGS.gui,
        record=False,
        obstacles=False,
        user_debug_gui=False,
    )

    #### Initialize the logger #################################
    logger = Logger(
        logging_freq_hz=int(ARGS.log_freq_hz / AGGR_PHY_STEPS),
        num_drones=ARGS.num_drones,
        duration_sec=int(ARGS.duration_sec),
        control_length=4
    ) # TODO correct control length
    LOG_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ / ARGS.log_freq_hz))

    #### Initialize the controllers ############################
    ctrl = [INDIControl(drone_model=drone) for drone in env.DRONE_MODEL]
    commands = [ np.zeros(4) for i in range(ARGS.num_drones) ]
    action = {str(i): np.zeros(4) for i in range(ARGS.num_drones)}
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ / ARGS.control_freq_hz))
    desired_course = INIT_YAW
    speed_setpoint = ARGS.speed_setpoint
    obs, _, _, _ = env.step(action) # initial observation

    #### Init SwarmFish ########################################
    params = load_params_from_yaml(ARGS.swarm_config)
    if ARGS.y_ali is not None:
        params.y_ali = ARGS.y_ali
    if ARGS.y_att is not None:
        params.y_att = ARGS.y_att

    #### Run the simulation ####################################
    START = time.time()
    for i in range(0, int(ARGS.duration_sec * env.SIM_FREQ), AGGR_PHY_STEPS):

        ### Compute action from commands ##########################
        for k, v in enumerate(commands):
            state = obs[str(k)]["state"]
            cmd, _, _ = ctrl[k].computeControl(
                control_timestep=env.AGGR_PHY_STEPS * env.TIMESTEP,
                cur_pos=state[0:3],
                cur_quat=state[3:7],
                cur_vel=state[10:13],
                cur_ang_vel=state[13:16],
                target_pos=state[0:3], # same as the current position
                target_rpy=np.array([0., 0., v[3]]),
                target_vel=v[0:3] # target the desired velocity vector
            )
            action[str(k)] = cmd
        #### Step the simulation ###################################
        obs, _, _, _ = env.step(action)

        #### Compute control at the desired frequency ##############
        if i % CTRL_EVERY_N_STEPS == 0:

            states = { str(j): sc.State(
                obs[str(j)]["state"][0:3],
                obs[str(j)]["state"][10:13],
                obs[str(j)]["state"][9], float(i / env.SIM_FREQ)) for j in range(ARGS.num_drones) }
            #### Compute control for the current state #############
            for uav_id in range(ARGS.num_drones):
                uav_name = str(uav_id)
                state = states[uav_name]
                neighbors = [ states[str(k)] for k in range(ARGS.num_drones) if k != uav_id ]
                cmd, influentials = sc.compute_interactions(state, params,
                        neighbors,
                        nb_influent=2,
                        wall=None,
                        altitude=None, z_min = 4., z_max = 8.,
                        direction=None)
                noise = np.random.normal(0., NOISES)
                desired_course[uav_id] = sc.wrap_to_pi(desired_course[uav_id] + cmd.delta_course / ARGS.control_freq_hz) + noise[0]
                velocity = speed_setpoint + cmd.delta_speed + noise[1] # TODO clip min/max
                vz = cmd.delta_vz + noise[2]
                speed = np.array([
                    velocity * math.cos(desired_course[uav_id]),
                    velocity * math.sin(desired_course[uav_id]),
                    vz,
                    desired_course[uav_id]])
                commands[uav_id] = speed


        #### Log the simulation ####################################
        if i % LOG_EVERY_N_STEPS == 0:
            for j in range(ARGS.num_drones):
                logger.log(drone=j,
                           timestamp=i/env.SIM_FREQ,
                           state=obs[str(j)]["state"],
                           control=commands[j]
                           )

        #### Sync the simulation ###################################
        if ARGS.gui:
            sync(i, START, env.TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    if ARGS.save_quant:
        quant = compute_quantification(logger.states[:,0:3,:], logger.states[:,10:13,:], int(logger.state_length * 0.2))
        file_name = os.path.join(ARGS.log_file_path, 'quant_'+ARGS.log_name+'.txt')
        #np.savez(file_name, quant=quant)
        np.savetxt(file_name, quant.reshape((1,6)))
    else:
        logger.save(file_path=ARGS.log_file_path, file_name=ARGS.log_name)

    #### Plot the simulation results ###########################
    # if ARGS.plot:
    # logger.plot()

    return f'{ARGS.log_file_path}{ARGS.log_name} done'

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(
        description="Phase diagram simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--drone", default=["robobee"], type=list, help="Drone model", metavar="", choices=[DroneModel])
    parser.add_argument("--num_drones", default=5, type=int, help="Number of drones", metavar="")
    parser.add_argument("--speed_setpoint", default=1., type=float, help="speed setpoint", metavar="")
    parser.add_argument("--physics", default="pyb", type=Physics, help="Physics updates", metavar="", choices=Physics)
    parser.add_argument("--gui", default=False, type=str2bool, help="Whether to use PyBullet GUI", metavar="")
    parser.add_argument("--plot", default=True, type=str2bool, help="Whether to plot the simulation results", metavar="")
    parser.add_argument("--aggregate", default=True, type=str2bool, help="Whether to aggregate physics steps", metavar="")
    parser.add_argument("--simulation_freq_hz", default=100, type=int, help="Simulation frequency in Hz", metavar="")
    parser.add_argument("--control_freq_hz", default=100, type=int, help="Control frequency in Hz", metavar="")
    parser.add_argument("--log_freq_hz", default=10, type=int, help="Log frequency in Hz", metavar="")
    parser.add_argument("--log_name", default=None, type=str, help="Log file name", metavar="")
    parser.add_argument("--log_file_path", default='logs/', type=str, help="Log file path", metavar="")
    parser.add_argument("--duration_sec", default=20, type=int, help="Duration of the simulation in seconds", metavar="")
    parser.add_argument("--swarm_config", default='config/demo.yaml', type=str, help="SwarmFish parameter file", metavar="")
    parser.add_argument("--random_init",  default=True, type=str2bool, help="Randomize init (heading only)", metavar="")
    parser.add_argument("--y_att", default=0.5, type=float, help="attraction coefficient", metavar="")
    parser.add_argument("--y_ali", default=0.5, type=float, help="aligment coefficient", metavar="")
    parser.add_argument("--save_quant", default=False, type=str2bool, help="Save quantification instead of full log", metavar="")
    ARGS = parser.parse_args()

    try:
        run_simulation(ARGS)
    except (KeyboardInterrupt, SystemExit):
        print("Simu phase diagram stopped")

# EOF

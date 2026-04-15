import yaml
import argparse
from swarmfish.swarm_control import SwarmParams, NavParams, ExploParams
import subprocess

def getSwarmParams(md_file_name: str) -> SwarmParams:
    swarm_params = SwarmParams(
    y_w=1.5,
    l_w=2.5,
    e1_w=1.0,
    e2_w=0.25,
    y_z_w=0.5,
    dz_w=1.,
    y_att=0.35,
    l_att=5.0,
    d0_att=3.,
    a_att=0.33,
    b1_att=0.,
    b2_att=0.,
    y_ali=0.2,
    l_ali=3.,
    d0_ali=2.,
    a_ali=0.,
    b1_ali=0.6,
    b2_ali=0.,
    y_acc=0.3,
    l_acc=2.5,
    d0_v=3.5,
    y_z=0.2,
    l_z=5.,
    a_z=1.,
    d0_z=2.,
    sigma_z=2.,
    y_nav=0.5,
    y_z_nav=0.5,
    y_vz_nav=0.1,
    y_intruder=1.2,
    y_z_intruder=1.,
    l_intruder=4.,
    y_obs=1.,
    t_obs=3.,
    e1_obs=1.4,
    e2_obs=0.,
    )

    with open(md_file_name, 'r') as md:
        md_lines = md.readlines()

        last_gen = md_lines[-3]
        last_gen.strip()

        params_string = last_gen.split(' | ')[3]
        params_string = params_string.strip('|\n')
        params_string = params_string.strip()

        params_list = params_string.split(', ')

        for param in params_list:
            param = param.split("=")

            name = param[0]
            value = float(param[1])

            if hasattr(swarm_params, name):
                setattr(swarm_params, name, value)

    return swarm_params

def getNavParams(md_file_name: str) -> NavParams:
    nav_params = NavParams(
        max_velocity=None,
        min_velocity=0.,
        zmax=None,
        zmin=None,
    )

    with open(md_file_name, 'r') as md:
        for line in md.readlines():

            if '|' in line:
                items = line.split('|')
                
                name = items[1].strip()
                value = items[2].strip()

                if name == 'MAX_SPEED':
                    setattr(nav_params, 'max_velocity', float(value))
                if name == 'Z_MIN':
                    setattr(nav_params, 'zmin', float(value))
                if name == 'Z_MAX':
                    setattr(nav_params, 'zmax', float(value))

    return nav_params

def getExploParams(md_file_name: str) -> ExploParams:
    explo_params = ExploParams(
        arena_radius=10.0,
        grid_res=1.0,
        sim_steps=100,
        neighbors=1
        )

    with open(md_file_name, 'r') as md:
        for line in md.readlines():

            if '|' in line:
                items = line.split('|')
                
                name = items[1].strip()
                value = items[2].strip()

                if name == 'ARENA_RADIUS':
                    explo_params.arena_radius = float(value)
                elif name == 'GRID_RES':
                    explo_params.grid_res = float(value)
                elif name == 'SIM_STEPS':
                    explo_params.sim_steps = int(value)
                elif name == 'NEIGHBORS':
                    explo_params.neighbors = int(value)

    return explo_params


def getNumDrones(md_file_name: str) -> int:
    numDrones = 1
    with open(md_file_name, 'r') as md:
        for line in md.readlines():

            if '|' in line:
                items = line.split('|')
                
                name = items[1].strip()
                value = items[2].strip()

                if name == 'NB_DRONES':
                    numDrones = int(value)

    return numDrones

def main(md_file_name: str, output_yaml_name: str):
    swarm_params = getSwarmParams(md_file_name)
    nav_params = getNavParams(md_file_name)
    explo_params = getExploParams(md_file_name)
    yaml_structure = {
            "Agent": {
                "SwarmParams": vars(swarm_params),
                "NavParams": vars(nav_params),
                "ExploParams": vars(explo_params)
            }
        }

    with open(output_yaml_name, 'w') as yaml_file:
        yaml.dump(yaml_structure, yaml_file, default_flow_style=False, sort_keys=False)


    return getNumDrones(output_yaml_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SwarmSim MD reports to YAML config.")
    parser.add_argument(
        "md_file", 
        type=str, 
        help="The file path to the input markdown (.md) report."
    )
    args = parser.parse_args()

    output_yaml_name = f"./config/{args.md_file.split('/')[-2]}.yaml"
    main(args.md_file, output_yaml_name)
    numDrones = getNumDrones(args.md_file)

    cmd = f"python exploration_simple_sim.py --swarm_config {output_yaml_name} --num_drones {numDrones}"
    subprocess.run(cmd, shell=True)

from swarmfish.swarm_control import SwarmParams, NavParams

MD_FILE_NAME = "../simpleSwarmSim/logs/2026-03-27_16-09-17_PyTorch_CUDA_3D/report_3D.md"


def generate_SwarmParams(md_file_name: str):
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

#TODO: Refactor the open() out of the function
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

def generate_NavParams(md_file_name: str):
    nav_params = NavParams(
        max_velocity=None,
        min_velocity=0.,
        zmax=None,
        zmin=None,
    )

#TODO: Refactor the open() out of the function
    with open(md_file_name, 'r') as md:
        md_lines = md.readlines()

        config_lines = md_lines[6:29]

        for line in config_lines:
            items = line.split(' | ')
            # print(items)
            name = items[0]
            value = items[1]

            name = name.strip('|').strip()
            value = value.strip().rstrip('|').strip()

            print(f'name: {name}, value: {value}')

            if name == 'MAX_SPEED':
                setattr(nav_params, 'max_velocity', float(value))
            if name == 'Z_MIN':
                setattr(nav_params, 'zmin', float(value))
            if name == 'Z_MAX':
                setattr(nav_params, 'zmax', float(value))

    return nav_params


if __name__ == "__main__":
    test = generate_SwarmParams(MD_FILE_NAME)
    test1 = generate_NavParams(MD_FILE_NAME)
    print(test1)


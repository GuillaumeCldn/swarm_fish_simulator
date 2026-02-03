import sys
import os
from multiprocess import Pool, freeze_support
import subprocess
import numpy as np
import json
import datetime
import argparse

def run_cmd(param: dict, sim_args):
    start_sim = datetime.datetime.now()
    current_env = os.environ.copy()
    
    # Calcul dynamique du chemin vers UavSwarmFish
    script_dir = os.path.dirname(os.path.abspath(__file__))
    swarmfish_path = os.path.abspath(os.path.join(script_dir, "..", "UavSwarmFish"))
    
    # Correction du PYTHONPATH pour Windows
    existing_pp = current_env.get("PYTHONPATH", "")
    pp_list = [swarmfish_path]
    if existing_pp:
        pp_list.extend(existing_pp.replace(':', ';').split(';'))
    
    unique_pp = []
    for p_path in pp_list:
        p_clean = p_path.strip()
        if p_clean and p_clean not in unique_pp:
            unique_pp.append(p_clean)
    current_env["PYTHONPATH"] = ";".join(unique_pp)

    cmd = [
        sys.executable,
        sim_args['sim'].strip(),
        f'--swarm_config={sim_args["swarm_config"].strip()}',
        f'--speed_setpoint={sim_args["speed"]}',
        f'--num_drones={sim_args["num_drones"]}',
        f'--duration_sec={sim_args["duration"]}',
        f'--log_name={param["name"].strip()}',
        f'--log_file_path={sim_args["log_path"].strip()}',
        f'--y_att={param["y_att"]}',
        f'--y_ali={param["y_ali"]}',
        '--random_init'
    ]
    
    if sim_args['quant']:
        cmd.append('--save_quant')

    # Nettoyage des arguments pour éviter les erreurs "unrecognized arguments"
    cmd = [str(c) for c in cmd if str(c).strip()]

    subprocess.run(cmd, env=current_env)
    print(f'Done [{param["count"]}] in [{datetime.datetime.now()-start_sim}]: {" ".join(cmd)}')

def main():
    parser = argparse.ArgumentParser(
        description="Build phase diagrams",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("log_path", type=str, help="log files path")
    parser.add_argument("swarm_config", type=str, help="swarm config yaml file")
    parser.add_argument("--config", type=str, default="config.json", help="json config file")
    parser.add_argument("--sim", type=str, default="simu_phase_diagram_simple_sim.py", help="simulator")
    parser.add_argument("--pool_size", type=int, default=4, help="job pool size")
    parser.add_argument("--quant", action='store_true', help="only save quantification")
    parser.add_argument("--num_drones", type=int, default=5, help="number of drones")
    parser.add_argument("--duration", type=int, default=180, help="duration in seconds")
    parser.add_argument("--samples", type=int, default=10, help="number of samples")
    parser.add_argument("--speed", type=float, default=1., help="drones flight speed")
    parser.add_argument("--y_att_max", type=float, default=2.)
    parser.add_argument("--y_ali_max", type=float, default=1.)
    parser.add_argument("--y_att_step", type=float, default=0.05)
    parser.add_argument("--y_ali_step", type=float, default=0.05)
    args = parser.parse_args()

    config = {
        'num_drones': args.num_drones,
        'duration': args.duration,
        'samples': args.samples,
        'speed': args.speed,
        'y_att': [0., args.y_att_max, args.y_att_step],
        'y_ali': [0., args.y_ali_max, args.y_ali_step]
    }

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    config_file = os.path.join(args.log_path, args.config)
    with open(config_file, 'w') as f:
        json.dump(config, f)

    y_atts = np.arange(config['y_att'][0], config['y_att'][1], config['y_att'][2])
    y_alis = np.arange(config['y_ali'][0], config['y_ali'][1], config['y_ali'][2])

    nb_run = len(y_atts) * len(y_alis) * args.samples
    idx = 0
    params = []
    for i, y_att in enumerate(y_atts):
        for j, y_ali in enumerate(y_alis):
            for k in range(args.samples):
                idx += 1
                params.append({
                    'name': f'run__{i}_{j}_{k}',
                    'y_att': y_att,
                    'y_ali': y_ali,
                    'count': f'{idx} / {nb_run}'
                })

    sim_args = vars(args)
    start_time = datetime.datetime.now()
    print(f'Starting {nb_run} simulations at time: {start_time}')

    p = None
    try:
        p = Pool(args.pool_size)
        # On utilise une fonction lambda ou wrapper pour passer les arguments fixes
        p.starmap(run_cmd, [(param, sim_args) for param in params])
    except (KeyboardInterrupt, SystemExit):
        print("\nArrêt manuel.")
    finally:
        if p is not None:
            p.close()
            p.join()

    end_time = datetime.datetime.now()
    print(f'End simulations at time: {end_time}, in {end_time - start_time}')

if __name__ == '__main__':
    freeze_support()
    main()
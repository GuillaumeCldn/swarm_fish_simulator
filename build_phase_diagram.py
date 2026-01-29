import sys
import os
from multiprocess import Pool
import subprocess
import numpy as np
import json
import datetime
import argparse

parser = argparse.ArgumentParser(
    description="Build phase diagrams",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("log_path", type=str, help="log files path")
parser.add_argument("swarm_config", type=str, help="swarm config yaml file")
parser.add_argument("--config", type=str, default="config.json", help="json config file")
parser.add_argument("--sim", type=str, default="simu_phase_diagram_simple_sim.py", help="simulator (python file name)")
parser.add_argument("--pool_size", type=int, default=4, help="job pool size")
parser.add_argument("--quant", action='store_true', help="only save quantification, no full logs")
parser.add_argument("--num_drones", type=int, default=5, help="number of drones")
parser.add_argument("--duration", type=int, default=180, help="duration in seconds")
parser.add_argument("--samples", type=int, default=10, help="number of samples")
parser.add_argument("--speed", type=float, default=1., help="drones flight speed")
parser.add_argument("--y_att_max", type=float, default=2., help="max value for y_ali")
parser.add_argument("--y_ali_max", type=float, default=1., help="max value for y_ali")
parser.add_argument("--y_att_step", type=float, default=0.05, help="step for y_att")
parser.add_argument("--y_ali_step", type=float, default=0.05, help="step for y_ali")
args = parser.parse_args()

config = {
        'num_drones': args.num_drones,
        'duration': args.duration,
        'samples': args.samples,
        'speed': args.speed,
        'y_att': [0., args.y_att_max, args.y_att_step],
        'y_ali': [0., args.y_ali_max, args.y_ali_step]
        }

config_file = os.path.join(args.log_path, args.config)
with open(config_file, 'w') as f:
    json.dump(config, f)

if args.quant:
    quant_opt = '--save_quant'
else:
    quant_opt = ''

def run_cmd(param: dict):
    start_sim = datetime.datetime.now()
    cmd = ['python', f'{args.sim}',
            f'--swarm_config={args.swarm_config}',
            f'--speed_setpoint={args.speed}',
            f'--num_drones={args.num_drones}',
            f'--duration_sec={args.duration}',
            f'--log_name={param["name"]}',
            f'--log_file_path={args.log_path}',
            f'--y_att={param["y_att"]}',
            f'--y_ali={param["y_ali"]}',
            '--random_init',
            quant_opt
            ]
    subprocess.run(cmd)
    print(f'Done [{param["count"]}] in [{datetime.datetime.now()-start_sim}]: {" ".join(cmd)}')

y_atts = np.arange(config['y_att'][0], config['y_att'][1], config['y_att'][2])
y_alis = np.arange(config['y_ali'][0], config['y_ali'][1], config['y_ali'][2])

nb_run = len(y_atts)*len(y_alis)*args.samples
idx = 0
params = []
for i, y_att in enumerate(y_atts):
    for j, y_ali in enumerate(y_alis):
        for k in range(args.samples):
            idx += 1
            param = {
                    'name': f'run__{i}_{j}_{k}',
                    'y_att': y_att,
                    'y_ali': y_ali,
                    'count': f'{idx} / {nb_run}'
                    }
            params.append(param)

start_time = datetime.datetime.now()
print(f'Starting {nb_run} simulations at time: {start_time}')

try:
    p = Pool(args.pool_size)
    p.map(run_cmd, params)
except (KeyboardInterrupt, SystemExit):
    print("build_phase_diagram stopped by hand")
finally:
    p.close()

end_time = datetime.datetime.now()
print(f'End simulations at time: {end_time}, in {end_time - start_time}')


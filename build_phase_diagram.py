import sys
import os
from multiprocess import Pool
import subprocess
import numpy as np
import json

#N = 3
pool_size = 20
num_drones = 5
duration = 200
samples = 10
speed = 1.
log_file_path = 'test_phase/'


p = Pool(pool_size)

config = {
        'num_drones': num_drones,
        'duration': duration,
        'samples': samples,
        'speed': speed,
        'y_att': [0., 1.0, 0.05],
        'y_ali': [0., 0.4, 0.05]
        }

config_file = os.path.join(log_file_path, 'config.json')
with open(config_file, 'w') as f:
    json.dump(config, f)

def run_cmd(param: dict):
    cmd = ['python', 'simu_phase_diagram.py',
            '--swarm_config=../UavSwarmFish/config/demo.yaml',
            f'--num_drones={num_drones}', f'--duration_sec={duration}',
            f'--log_name={param["name"]}',
            f'--log_file_path={log_file_path}',
            f'--y_att={param["y_att"]}',
            f'--y_ali={param["y_ali"]}'
            ]
    subprocess.run(cmd)
    print("Done:",cmd)

y_atts = np.arange(config['y_att'][0], config['y_att'][1], config['y_att'][2])
y_alis = np.arange(config['y_ali'][0], config['y_ali'][1], config['y_ali'][2])

params = []
for i, y_att in enumerate(y_atts):
    for j, y_ali in enumerate(y_alis):
        for k in range(samples):
            param = {
                    'name': f'run__{i}_{j}_{k}',
                    'y_att': y_att,
                    'y_ali': y_ali,
                    }
            params.append(param)

p.map(run_cmd, params)
p.close()


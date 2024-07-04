import json
import sys
sys.path.append('/home/gautier/dev/swarm/UavSwarmFish')
from os import path

import argparse
import numpy as np
import math
import matplotlib.pyplot as plt

from swarmfish.utils import compute_quantification_from_log

parser = argparse.ArgumentParser(
    description="Phase diagram simulation",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("log_path", type=str, help="log files path")
parser.add_argument("--config", type=str, default="config.json", help="json config file")

args = parser.parse_args()

f = open(path.join(args.log_path, args.config), 'r')
config = json.load(f)
f.close()

y_atts = np.arange(config['y_att'][0], config['y_att'][1], config['y_att'][2])
y_alis = np.arange(config['y_ali'][0], config['y_ali'][1], config['y_ali'][2])
samples = config['samples']

# array to store quantification (6 values)
data = np.full((y_atts.shape[0], y_alis.shape[0], samples, 6), np.nan)

for i, y_att in enumerate(y_atts):
    for j, y_ali in enumerate(y_alis):
        for k in range(samples):
            file_name = path.join(args.log_path, f'run__{i}_{j}_{k}.npy')
            try:
                quant = compute_quantification_from_log(file_name)
                data[i,j,k,:] = quant
            except Exception as e:
                pass

fig, axs = plt.subplots(2, 3)
x,y = np.meshgrid(y_atts, y_alis)
dispersion      = np.nanmean(data[:,:,:,0], axis=2)
polarization    = np.nanmean(data[:,:,:,1], axis=2)
milling         = np.nanmean(data[:,:,:,2], axis=2)
fluct_disp      = np.nanmean(data[:,:,:,3], axis=2)
fluct_pol       = np.nanmean(data[:,:,:,4], axis=2)
fluct_mil       = np.nanmean(data[:,:,:,5], axis=2)
axs[0, 0].pcolormesh(x, y, np.transpose(dispersion))
axs[0, 0].set_title('dispersion')
axs[0, 1].pcolormesh(x, y, np.transpose(polarization))
axs[0, 1].set_title('polarization')
axs[0, 2].pcolormesh(x, y, np.transpose(milling))
axs[0, 2].set_title('milling')
axs[1, 0].pcolormesh(x, y, np.transpose(fluct_disp))
axs[1, 0].set_title('fluctuation of dispersion')
axs[1, 1].pcolormesh(x, y, np.transpose(fluct_pol))
axs[1, 1].set_title('fluctuation of polarization')
axs[1, 2].pcolormesh(x, y, np.transpose(fluct_mil))
axs[1, 2].set_title('fluctuation of milling')

axs[0, 0].set_ylabel(r'$\gamma_{ali}$')
axs[1, 0].set_ylabel(r'$\gamma_{ali}$')
axs[1, 0].set_xlabel(r'$\gamma_{att}$')
axs[1, 1].set_xlabel(r'$\gamma_{att}$')
axs[1, 2].set_xlabel(r'$\gamma_{att}$')
plt.show()

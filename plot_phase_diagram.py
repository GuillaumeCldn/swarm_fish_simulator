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
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True, help="display result in a plot")
parser.add_argument("--quant", action=argparse.BooleanOptionalAction, default=False, help="load from quantification logs")
parser.add_argument("--save", type=str, default=None, help="save in file")
parser.add_argument("--load", type=str, default=None, help="load data from file")

args = parser.parse_args()

if args.load is None:
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
                if args.quant:
                    file_name = path.join(args.log_path, f'quant_run__{i}_{j}_{k}.txt')
                    try:
                        quant = np.loadtxt(file_name)
                        data[i,j,k,:] = quant
                    except Exception as e:
                        print(e)
                else:
                    file_name = path.join(args.log_path, f'run__{i}_{j}_{k}.npy')
                    try:
                        quant = compute_quantification_from_log(file_name, skip_ratio=0.1)
                        data[i,j,k,:] = quant
                    except Exception as e:
                        pass

    dispersion      = np.nanmean(data[:,:,:,0], axis=2)
    polarization    = np.nanmean(data[:,:,:,1], axis=2)
    milling         = np.nanmean(data[:,:,:,2], axis=2)
    fluct_disp      = np.nanmean(data[:,:,:,3], axis=2)
    fluct_pol       = np.nanmean(data[:,:,:,4], axis=2)
    fluct_mil       = np.nanmean(data[:,:,:,5], axis=2)

else:
    data = np.load(args.load)
    config = data['config']
    dispersion = data['dispersion']
    polarization = data['polarization']
    milling = data['milling']
    fluct_disp = data['fluct_disp']
    fluct_pol = data['fluct_pol']
    fluct_mil = data['fluct_mil']

    y_atts = np.arange(config['y_att'][0], config['y_att'][1], config['y_att'][2])
    y_alis = np.arange(config['y_ali'][0], config['y_ali'][1], config['y_ali'][2])
    samples = config['samples']

if args.save is not None:
    np.savez(args.save,
            config=config,
            dispersion=dispersion,
            polarization=polarization,
            milling=milling,
            fluct_disp=fluct_disp,
            fluct_pol=fluct_pol,
            fluct_mil=fluct_mil)

if args.plot:
    fig, axs = plt.subplots(2, 3)
    x,y = np.meshgrid(y_atts, y_alis)
    p00 = axs[0, 0].pcolormesh(x, y, np.transpose(dispersion))
    axs[0, 0].set_title('dispersion')
    p01 = axs[0, 1].pcolormesh(x, y, np.transpose(polarization))
    axs[0, 1].set_title('polarization')
    p02 = axs[0, 2].pcolormesh(x, y, np.transpose(milling))
    axs[0, 2].set_title('milling')
    p10 = axs[1, 0].pcolormesh(x, y, np.transpose(fluct_disp))
    axs[1, 0].set_title('fluctuation of dispersion')
    p11 = axs[1, 1].pcolormesh(x, y, np.transpose(fluct_pol))
    axs[1, 1].set_title('fluctuation of polarization')
    p12 = axs[1, 2].pcolormesh(x, y, np.transpose(fluct_mil))
    axs[1, 2].set_title('fluctuation of milling')

    axs[0, 0].set_ylabel(r'$\gamma_{ali}$')
    axs[1, 0].set_ylabel(r'$\gamma_{ali}$')
    axs[1, 0].set_xlabel(r'$\gamma_{att}$')
    axs[1, 1].set_xlabel(r'$\gamma_{att}$')
    axs[1, 2].set_xlabel(r'$\gamma_{att}$')

    fig.colorbar(p00, ax=axs[0,0], pad=0)
    fig.colorbar(p01, ax=axs[0,1], pad=0)
    fig.colorbar(p02, ax=axs[0,2], pad=0)
    fig.colorbar(p10, ax=axs[1,0], pad=0)
    fig.colorbar(p11, ax=axs[1,1], pad=0)
    fig.colorbar(p12, ax=axs[1,2], pad=0)
    plt.show()

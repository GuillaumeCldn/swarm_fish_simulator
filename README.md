# UAV swarm simulator

## Installation 

Clone this repository in a folder adjacent to the folder containing the [UavSwarmFish](https://github.com/GuillaumeCldn/UavSwarmFish) repository:

```text
Project Root
├── UavSwarmFish/
└── swarm_fish_simulator/
```

Initialise a python virtual environment in `./swam_fish_simulator/` and download packages in `requirements.txt`.

In `./swarm_fish_simulator/bin/activate` export the path to `./UavSwarmFish/`to `$PYTHONPATH`.

Activate the virtual environment.


## Scripts

### build_phase_diagram.py

This script runs simulations in parallel. 

FIXME: When running simulations  with defaults, unrecognized argument errors are systematically raised for `simu_phase_diagram_simple_sim.py` but the simulations seem to still run. There might also be a problem with the logs not being created.

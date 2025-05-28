# Copyright (c) 2025, ETH Zurich
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import h5py
rave_sim_dir = Path('path/to/pyjton/python_packages/rave-sim')
simulations_dir = Path('path/to/data/')
scratch_dir = simulations_dir
sys.path.insert(0, str(rave_sim_dir / "big-wave"))
import multisim
import config
import util
from nist_lookup.xraydb_plugin import xray_delta_beta
# constants
h = 6.62607004 * 10**(-34) # planck constant in mˆ2 kg / s
c_0 = 299792458 # speed of light in m / s
eV_to_joule = 1.602176634*10**(-19)
N_A = 6.02214086 * 10**23 #[1/mol]


# constraints:
design_energy = 57000.0
thick = h * c_0 / (design_energy*eV_to_joule)/(2*(xray_delta_beta("Au", 19.32, design_energy)[0]-xray_delta_beta("Si", 2.34, design_energy)[0]))
g0_offset = 0.1
s=1.1-g0_offset
d1 = 0.2
d2=s-d1
pf = 40e-6
lambda_ = h * c_0 / (design_energy*eV_to_joule)
p0 = d1/d2*pf
acl= lambda_*(d2/2)/pf
P = d1/s*pf
print("autocorrelation length: ", acl)

M = (d1+d2)/d1
print(p0,P,pf)

config_dict = {
        "sim_params": {
            "N": 2**29,
            "dx": 0.2e-10,
            "z_detector": s,
            "detector_size": 2e-3,
            "detector_pixel_size_x": 1e-6,
            "detector_pixel_size_y": 1,
            "chunk_size": 256 * 1024 * 1024 // 16,  # use 256MB chunks
        },
        "dtype": "c8",
        "use_disk_vector": False,
        "save_final_u_vectors": False,
        "multisource": {
            "type": "points",
            "energy_range": [10000.0, 100000.0],
            "x_range": [-0e-6, 0e-6],
            "z": 0.0,
            "nr_source_points": 2000,
            "seed": 1,
        },
        "elements": [
            {
                "type": "grating",
                "pitch": float(P),
                "dc": [0.5, 0.5],
                "z_start": float(d1),#+0.1,
                "thickness": float(300e-6),
                "nr_steps": 50,
                "x_positions": [0.0],
                "substrate_thickness": (500) * 1e-6 - float(300e-6),
                "mat_a": ["Si", 2.34],
                "mat_b": ["Au", 19.32],
                "mat_substrate": ["Si", 2.34],
            },
        ],
    }
sim_path_att = multisim.setup_simulation(config_dict, Path("."), simulations_dir)
for i in range(config_dict["multisource"]["nr_source_points"]):
    os.system(f"CUDA_VISIBLE_DEVICES=0 path/to/python/python_packages/rave-sim/fast-wave/build-Release/fastwave -s {i} {sim_path_att}")

print(sim_path_att)
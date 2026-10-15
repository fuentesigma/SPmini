#
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar

# Radial profiles (code units) of the snapshots of a corenfw run, used by figure_q2
# usage: python export_profile.py results/corenfw/iso_hpc/n384

import os
import sys
import glob
import json

import numpy as np
import torch

import halo

folder = sys.argv[1]
L = json.load(open(os.path.join(folder, "settings.json")))["L"]
out = {}
for f in sorted(glob.glob(os.path.join(folder, "psi_t*.npz"))):
    psi = np.load(f)["psi"]
    n = psi.shape[0]
    dx = L / n
    rho = np.abs(psi) ** 2
    cell = np.unravel_index(np.argmax(rho), rho.shape)
    centre = halo.peak(torch.fft.fftn(torch.tensor(psi)), L, (np.array(cell) + 0.5) * dx - L / 2)
    r_cells = halo.periodic_distance(n, L, centre)
    _, d, r = halo.shells(rho, r_cells, dx, np.geomspace(dx / 2, L / 2, 40))
    tag = os.path.basename(f)[4:-4]
    out[f"r_{tag}"], out[f"rho_{tag}"] = r, d
    print(tag, flush=True)
np.savez(os.path.join(folder, "profiles_code_units.npz"), dx=dx, L=L, **out)

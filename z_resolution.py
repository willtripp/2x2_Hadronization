"""
Created by W. M. Tripp on November 5th 2026

How well could we possibly reconstruct z?

1. Select out interactions with containd HADRONS...
    Should I include pi-zeros?

2. Sum up all the visible hadronic energy....

3. Take highest energy hadron...

4. Would also be good to know how many events have only one uncontained hadron... This would bias our data

plots:
1) truth and reco z distribution
2) z resolution
3) hadronic energy recolution:
    Both using info available to the reco (so not neutros, etc...) and using the neutrino + lepton energy....
4) leading hadron energy resolution
"""

import numpy as np
import pandas as pd
from numba.core.cgutils import false_bit

from plot_data_mc import plot_data_mc
from plot_data_mc import plot_2d_hist

from SPINE_Reader import load_driver

def get_reco_z(reco_interaction):
    particles = reco_interaction.particles

    hadronic_energy = 0.0
    max_E           = 0.0

    for p in particles:
        if p.shape != 1:
            continue
        if p.pid == 2:
            continue

        #grab whatever spine thinks is the best ke (either range or calo)
        ke   = p.ke
        mass = p.mass

        E = np.sqrt(mass ** 2 + ke ** 2)

        if E > max_e:
            max_E = E
        hadronic_energy += E

    z = max_E / hadronic_energy
    return z

def get_truth_z(truth_interaction):
    W = truth_interaction.hadronic_invariant_mass
    x = truth_interaction.bjorkenx
    y = truth_interaction.inelasticity
    energy_transfer = truth_interaction.energy_transfer

    max_E = 0.0
    for p in truth_interaction.particles:
        if p.shape != 1:
            continue
        if p.pid == 2:
            continue
        E = p.energy_init
        if E > max_E:
            max_E = E

    z = max_E / energy_transfer

    return z

def get_containment(interaction):
    '''
    Function to decide if tha hadrons are contained or not. We don't care about anything else
    '''
    particles = interaction.particles
    for p in particles:
        if p.shape != 1:
            continue
        elif p.pid == 2:
            continue
        elif p.is_contained == False:
            return False
        else:
            return True

MC_driver = load_driver("/Users/williamtripp/Desktop/DUNE/simulation/MiniRun6.5/SPINE/MiniRun6.5_1E19_RHC.spine.0000000.MLRECO_SPINE.hdf5")

entry_list   = []
reco_z  = []
truth_z = []

for entry in range(len(MC_driver)):
    mc = MC_driver.process(entry)
    pairs = mc["interaction_matches_r2t"]
    overlaps = mc["interaction_matches_r2t_overlap"]

    entry_list.append(entry)

    for i, (reco_int, truth_int) in enumerate(pairs):
        if overlaps[i] < 0.5:
            continue
        #fixme: could be one line
        if get_containment(reco_int) = False:
            continue
        if get_containment(truth_int) = False:
            continue

        reco_z.append(get_reco_z(reco_int))
        truth_z.append(get_truth_z(truth_int))


resolution = np.asarray(reco_z) - np.asarray(truth_z)



df = pd.DataFrame({"entry": entry_list, "reco_z": reco_z, "truth_z": truth_z, "resolution": resolution})

plot_data_mc(

    column="thickness",
    bins=np.linspace(0, 5, 51),
    xlabel="Track thickness [size / length]",
    title="Track thickness: Data vs MC",
    normalize_mc_to_data=True,
    outfile="thickness_data_mc.png",
)

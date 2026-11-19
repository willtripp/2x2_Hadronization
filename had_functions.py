"""
Created on November 19th, 2026 by W. M. Tripp

This is a place to consolidate the functions I'm using for hadronization

It will also serve as a good place to start some larger alterations now that I've got preleminary results
"""


'''
##### ----- Reconstruction Based Functions ----- #####
'''

import numpy as np

# def had_identification(reco_interaction):
#     '''
#     I may want to augment this function to only include the longest track as the muon (though
#     '''


# def reco_had_containment(reco_int, charged_only=True):
#     '''
#     This function takes a reco interaction and tells us if the relevant particles are contained
#     There will be two methods, one to make sure all non-muons are contained and another to make sure only
#     track like hadrons are contained
#     '''
#
#     particles = reco_int['reco_particles']
#     lengths   = []
#
#     if chargd_only == False:
#         for p in particles:
#             if p.shape == 2
#
#     # fixme: need a mechanism for selecting the longest track...
#     for p in particles:
#         if p.shape == 1:
#             lengths.append([p.length])
#
#
# def reco_energy_trans_est(reco_int):



'''
##### ----- Truth Based Funcitons ----- #####
'''

def truth_containment(truth_interaction, charged_only= True):

    particles = truth_interaction.particles

    if charged_only:
        #In this case we want all charged hadrons to be contained
        for p in particles:
            if p.shape == 1:
                if p.pid == 2:
                    continue
                if p.is_contained == False:
                    return False
        return True
    else:
        for p in particles:
            if p.shape == 2 or p.shape == 3:
                continue
            if p.is_contained == False:
                return False
        return True





'''
##### ----- RecoTruth Matched Functions ----- #####
'''

def get_best_pid(reco_particle, truth_particles):
    matches = np.asarray(reco_particle.match_ids)
    overlaps = np.asarray(reco_particle.match_overlaps)

    best_overlap = np.max(overlaps)

    if best_overlap < 0.5:
        return reco_particle.pid

    mask = overlaps == best_overlap

    best_match = matches[mask]

    if len(best_match) != 1:
        raise ValueError ("There are multiple best matches!?!?")

    return truth_particles[best_match[0]].pid

def get_primary(reco_particle, truth_particles):
    matches = np.asarray(reco_particle.match_ids)
    overlaps = np.asarray(reco_particle.match_overlaps)

    best_overlap = np.max(overlaps)
    if best_overlap < 0.5:
        return reco_particle.is_primary
    else:
        mask = overlaps == best_overlap
        best_match = matches[mask]

        return truth_particles[best_match[0]].is_primary


#
# def pseudo_reco_energy_trans_est(reco_int, truth_int):
#
#
#
# def n_pi_cut(interaction):

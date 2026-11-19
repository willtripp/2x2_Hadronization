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


def est_transferred_energy(interaction):
    #TODO: figure out if we should include electrons here to get a better handle on transferred energy rather than ehadvis
    e_trans = 0.0
    particles = interaction.particles
    for p in particles:
        #skip leptons
        if p.pid == 1 or p.pid ==2:
            continue
        # assume photons come form hadrons (or at least transferred energy)
        if p.pid == 0:
            e_trans += p.ke
        # assume protons already existed
        if p.pid == 4:
            e_trans += p.ke
        # assume everything else was created by interaction
        else:
            e_trans += np.sqrt(p.mass ** 2 + p.p ** 2)
    # print(e_trans)

    return e_trans



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

    best_pid = truth_particles[best_match[0]].pid

    if best_pid == -1:
        return reco_particle.pid
    else:
        return best_pid

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



def pseudo_reco_est_transferred_energy(reco_int, truth_int):
    e_trans = 0.0
    particles = reco_int.reco_particles

    #particle masses in MeV
    mpi = 139.57
    mk  = 493.68

    for p in particles:
        pid = get_best_pid(p, truth_int.truth_particles)

        #skip leptons
        if pid == 1 or pid == 2:
            continue
        #assume photons come from hadrons (or at least some transferred energy)
        if p.pid == 0:
            e_trans += p.ke
        #assume protons already existed in the nucleus
        if p.pid == 4:
            e_trans +=p.ke
        #pions
        if pid == 4:
            e_trans += (p.ke + mpi)
        #kaons
        if pid == 5:
            e_trans += (p.ke + mk)

    return e_trans


def get_pids(reco_int, truth_int):
    '''
    might be nice to do this once and feed the output rather than performing loop after loop that extracts the same info
    '''

    reco_particels = reco_int.reco_particles
    pids = []
    for p in reco_particels:
        pids.append(get_best_pid(p, truth_int.truth_particles))

    return pids




def n_pi_cut(reco_int, truth_int, n):
    '''
    Parameters
    ----------
    reco_int
    truth_int
    n: minimum number of pions for an event to pass this cut

    Returns: bool
    -------

    '''

    reco_particles = reco_int.reco_particles
    pids = []
    for p in reco_particles:
        pids.append(get_best_pid(p, truth_int.truth_particles))

    pids = np.asarray(pids)
    mask = pids == 3

    pions = pids[mask]

    if len(pions) >= n:
        return True
    else:
        return False

def has_mu(reco_int, truth_int):
    particles = reco_int.particles
    pids = []
    for p in particles:
        pids.append(get_best_pid(p, truth_int.truth_particles))

    pids = np.asarray(pids)
    mask = pids == 2

    muons = pids[mask]

    if len(muons) >= 1:
        return True

    else:
        return False

def get_reco_truth_z(reco_int, truth_int):

    #get the transferred energy
    approx_energy_transfer = pseudo_reco_est_transferred_energy(reco_int, truth_int)

    if approx_energy_transfer == 0.0:
        raise ValueError("There is no energy transfer, how did this event pass selection?")

    #find leading hadron

    #particle energies in MeV / c^2
    mpi   = 139.57
    mk    = 493.68
    mprot = 938.27

    max_had_E = 0.0

    particles = reco_int.reco_particles
    for p in particles:
        primary = get_primary(p, truth_int.truth_particles)
        if not primary:
            continue
        pid = get_best_pid(p, truth_int.truth_particles)
        if pid == 1 or pid == 2:
            continue
        #pions
        if pid == 3:
            E = mpi + p.ke
        elif pid == 4:
            E = mprot + p.ke
        elif pid == 5:
            E = mprot + p.ke

        if E > max_had_E:
            max_had_E = E

    z = max_had_E / approx_energy_transfer
    return z
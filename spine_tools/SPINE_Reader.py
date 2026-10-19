'''
Created by W. M. Tripp on July 10th, 2026

The goal is to create a function that, given a SPINE driver and a set of base attributes returns .csv files containign
data frames of desired information.
'''

import csv
import gc
import time
import yaml

from pathlib import Path
from spine.driver import Driver

def load_driver(filename):
    '''
    Finally: a function that when given a file path, loads the driver for you so one doesn't have to write it out every
    time a new script is required
    '''
    #fixme: add in some detector geometry stuff
    DATA_PATH = filename
    CONFIG    = Path("read_spine_hdf5.yaml")
    cfg_txt   = CONFIG.read_text().replace("DATA_PATH", DATA_PATH)
    cfg       = yaml.safe_load(cfg_txt)

    return Driver(cfg)

def get_part_attr(particles, base_attributes):
    '''Take in a particle, set of attributes and a shape
    return a row of attributes for each particle'''
    for p in particles:
        # if getattr(p, "shape", -1) != shape:
        #     continue
        row = {}
        for a in base_attributes:
            row[a] = getattr(p, a, -1)
        row["thickness"] = row["size"] / row["length"]
        # fixme: num_fragments seems to always be zero...
        # row["fragment_density"] = row["num_fragments"] / row["length"]
        yield row

def iter_part_to_csv(driver, out_file, base_attributes, all_attributes, shape, truth=False):
    '''
    :param driver: Give it a spine driver to loop over
    :param out_file: set a output file name
    :param base_attributes: a set of base attirbutes that appear in the reco particle key
    :param all_attributes:  full list of attributes including calculated ones
    :param shape: Give it a shape to include
    :param truth: Flag for filling a true particle spectra
    :return: a csv with all of the file info
    '''
    with open(out_file, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=all_attributes)
        writer.writeheader()

        # Loop over every entry in the driver
        for entry in range(len(driver)):
            data = driver.process(entry=entry)
            if truth:
                parts = data.get("truth_particles", [])
            else:
                parts = data.get("reco_particles", [])
            if len(parts) == 0:
                continue

            # Save rows to csv individually to save memory
            for row in get_part_attr(parts, base_attributes, shape):
                writer.writerow(row)

            # Delete File to save memory
            del data

            if entry == int( len(driver) / 4 ):
                print("Quarter of the way there!")
                gc.collect()
            if entry == int( len(driver) / 2 ):
                print("Halfway There!")
                gc.collect()

        print(f"Done Writing to {out_file}!")

def iter_part_speedup(driver, out_file, base_attributes, all_attributes):
    '''
    follows similar structure as above: Main difference is writing out in batches to reduce i/o usage for a speed benefit
    The vast majority of the time is from processing driver entries, I think my speedup journey ends here...
    '''
    n_entries    = len(driver)          #set number of entries to loop over
    progress_int = n_entries // 10      #set update frequency
    flush_int    = 7500                 #set particle batch size

    match_threshold = 0.666             #set overlap threshold -> add other cut vairable here if desired

    t0 = time.time()

    with open(out_file, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames = all_attributes)
        writer.writeheader()

        # fill up a batch and then write out in bulk to not slam i/o
        batch = []

        for entry in range(n_entries):
            data  = driver.process(entry)
            parts = data.get("reco_particles", [])

            for row in get_part_attr(parts, base_attributes):
                batch.append(row)

            if len(batch) >= flush_int:
                writer.writerows(batch)
                del batch
                gc.collect()
                batch = []

            if entry % progress_int == 0 and entry > 0:
                elapsed = time.time() - t0
                rate    = entry / elapsed
                eta     = (n_entries - entry) / rate if rate > 0 else float(inf)

                print(f"{100*entry/n_entries}% Complete")
                print(f"Rate: {rate:.2f} entries/s")
                print(f"ETA {eta/60:.1f} min")

        if batch:
            writer.writerows(batch)
            batch.clear()

def get_nonzero_entries(driver):
    '''
    From event display escalades, it appears that most of teh data entries are completely empty... What we want is to create
    a list of the non_empty entriers to greatly reduce the number of times driver.process needs to be calles in the future
    :param driver: A spine driver that you believe to have entries that are largely void of particles
    :return: a list of entry numbers that contain useful information
    '''

    n_entries  = len(driver)
    t0         = time.time()
    prog_int   = n_entries // 10
    entry_list = []

    for entry in range(n_entries):
        data = driver.process(entry=entry)
        part = data.get("reco_particles", [])
        if len(part) == 0:
            continue
        else:
            entry_list.append(entry)

        if entry % prog_int == 0 and entry > 0:
            elapsed = time.time() - t0
            rate    = entry / elapsed
            eta     = (n_entries - entry) / rate if rate > 0 else float(inf)

            print(f"{100*entry/n_entries}% Complete")
            print(f"Rate: {rate:.2f} entries/s")
            print(f"ETA {eta/60:.1f} min")

    print(f"Finished in a time of {(time.time() - t0):2f} seconds!")
    return entry_list

def reduced_data_loader(driver, out_file, entries_list, attributes):
    '''
    :param driver:       A SPINE driver to open up the events
    :param entries_list: A list of entries that are known to contain at least 1 particle
    :param attributes:   A list of attributes to grab from the particles
    :return:             This function will save to a csv file as it goes

    The goal here is to create a method to iterate over ONLY entries that contain information. This will reduce the number
    of calls to driver.Process by literally millions...
    '''

    # ----- Some Precursors to set batch sizes for memory/io efficiency
    n_entries     = len(entries_list)
    progress_int  = n_entries // 10
    progress      = 0
    batch_size    = 7500

    t0 = time.time()

    with open(out_file, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=attributes)
        writer.writeheader()

        batch = []

        # Iterate over entries in the list
        #fixme: the entries are limited for testing, must resolve
        for entry in entries_list:
            data      = driver.process(entry=entry)
            particles = data.get("reco_particles", [])

            # loop over particles in the entry and save their desired attributes
            for p in particles:
                row = {}
                for a in attributes:
                    row[a] = getattr(p, a, -1)
                row["entry"] = entry
                batch.append(row)

            # When the batches get large enough, write out info and flush the batch
            if len(batch) >= batch_size:
                writer.writerows(batch)
                batch.clear()
                gc.collect()

            progress += 1

            # Progress notifications
            if progress % progress_int == 0 and progress > 0:
                elapsed = time.time() - t0
                rate    = progress / elapsed
                eta     = (n_entries - progress) / rate if rate > 0 else float(inf)

                print(f"{100*progress/n_entries:.1f}% Complete")
                print(f"Rate: {rate:.2f} entries/s")
                print(f"ETA {eta/60:.1f} min")

        if batch:
            writer.writerows(batch)
            batch.clear()

def iter_trigger(driver, out_file, base_attributes, all_attributes):
    '''
    Adapted from iter_part_speedup... This takes the speedups we've learned above and applies a cut on trigger.
    Note that both beam and cosmic ["trigger"]["type"] 5 or 6 id accepted

    This still relies on processing the driver for each entry.. I think a useful combination of the above would be:
        1. Find all of the entries with triggers
    '''
    n_entries    = len(driver)          #set number of entries to loop over
    progress_int = n_entries // 10      #set update frequency
    flush_int    = 7500                 #set particle batch size

    match_threshold = 0.666             #set overlap threshold -> add other cut vairable here if desired

    t0 = time.time()

    with open(out_file, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames = all_attributes)
        writer.writeheader()

        # fill up a batch and then write out in bulk to not slam i/o
        batch = []

        for entry in range(n_entries):
            data      = driver.process(entry)
            trig_type = data["trigger"]["type"]
            if (trig_type != 5) && (trig_type != 6) :
                continue
            parts = data.get("reco_particles", [])

            for row in get_part_attr(parts, base_attributes):
                batch.append(row)

            if len(batch) >= flush_int:
                writer.writerows(batch)
                del batch
                gc.collect()
                batch = []

            if entry % progress_int == 0 and entry > 0:
                elapsed = time.time() - t0
                rate    = entry / elapsed
                eta     = (n_entries - entry) / rate if rate > 0 else float(inf)

                print(f"{100*entry/n_entries}% Complete")
                print(f"Rate: {rate:.2f} entries/s")
                print(f"ETA {eta/60:.1f} min")

        if batch:
            writer.writerows(batch)
            batch.clear()
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 12:34:23 2025

@author: YERAI
"""

from mod_construct_brick_Y import *
from mod_sample import *
from mod_construct_supercell_Y import *
from mod_write_Y import *
from mod_check import *
from parameters import *
from mod_make_graphs import *
import time
import sys

# Check input parameters:
try: seed
except NameError: seed = 1123

try: width_Ca_Si
except NameError: width_Ca_Si = 0.1

try: width_SiOH
except NameError: width_SiOH = 0.08

try: width_CaOH
except NameError: width_CaOH = 0.04

try: offset_gaussian
except NameError: offset_gaussian = False

try: make_independent
except NameError: make_independent = False

try: create
except NameError: create = False

try: check
except NameError: check = False

try: write_lammps
except NameError: write_lammps = True

try: write_lammps_erica
except NameError: write_lammps_erica = True

try: write_vasp
except NameError: write_vasp = False

try: write_siesta
except NameError: write_siesta = False

try: read_structure
except NameError: read_structure = False

try: surface_separation
except NameError: surface_separation = False

try: prefix
except NameError: prefix = "input"

try: substitute
except NameError: substitute=[0]

try: grid
except NameError: grid=[0]


widths = [width_Ca_Si, width_SiOH, width_CaOH]

np.random.seed(seed)
random.seed(seed+10)





if create or check:
    # Get all possible bricks
    bricks, sorted_bricks = get_all_bricks(pieces)
    #print(sorted_bricks)
    #sys.exit()
    # for CaSi in sorted_bricks:
    #     if 0 in sorted_bricks[CaSi]:
    #         for Si in sorted_bricks[CaSi][0]:
    #             for Ca in sorted_bricks[CaSi][0][Si]:
    #                 print(CaSi, Si, Ca, len(sorted_bricks[CaSi][0][Si][Ca]))
    #     print()


#combs = [ b.comb for b in bricks ]

#s = set(tuple(i) for i in combs)


if create or read_structure:
    if not os.path.isdir("./output_Y"):
        os.makedirs('./output_Y')


list_properties = []

if create:

    unitcell = np.array([ [6.7352,    0.0 ,      0.0],
                            [-4.071295, 6.209521,  0.0],
                           [0.7037701, -6.2095578, 13.9936836] ])

    supercell = np.zeros((3,3))
    for i in range(3):
        supercell[i,:] = unitcell[i,:]*size[i]



    N_brick = size[0]*size[1]*size[2]

    offset = [0.0, 0.0]
    if offset_gaussian:
        off_Si, off_Ca = get_offset(500, sorted_bricks, Ca_Si_ratio, W_Si_ratio, N_brick, widths)
        offset = [off_Si, off_Ca]


    list_crystals = []
    cont = 0


    t = time.time()
    jsample = 0
    for isample in range(N_samples):
        crystal, N_Ca, N_Si, r_SiOH, r_CaOH, MCL, N_water, r_2H_Si = sample_Ca_Si_ratio(
                                                    sorted_bricks, Ca_Si_ratio, W_Si_ratio, N_brick, widths, offset=offset )


        crystal_index = [ brick.ind for brick in crystal ] 

        new = False

        if make_independent and set(crystal_index) not in list_crystals:
            new = True
            list_crystals.append(set(crystal_index))
        elif not make_independent and crystal_index not in list_crystals:
            new = True
            list_crystals.append(crystal_index)

        if new:
            list_properties.append( [N_Ca/N_Si, r_SiOH, r_CaOH, MCL, jsample+1, r_2H_Si] )

            water_in_crystal = fill_water(crystal, N_water = N_water)

            crystal_rs, water_in_crystal_rs =  resize_crystal(crystal, water_in_crystal, size)
            entries_crystal, entries_bonds, crystal_dict, water_dict = get_full_coordinates( crystal_rs, water_in_crystal_rs, size, pieces, guest_ions, substitute )

            entries_angle = get_angles(crystal_dict, water_dict, size)

            # Water molecule overlap
            entries_crystal, N_not_ok, itry = check_move_water_hydrogens(entries_crystal)
            
            if N_not_ok != 0:
                print("Warning: structure {: 5d} contains {: 5d} wrong water molecules".format(jsample, N_not_ok))
            else:
                print("Structure {: 5d} converged after {: 5d} iterations".format(jsample, itry))


            write_output( jsample, entries_crystal, entries_bonds, entries_angle, size, crystal_rs, water_in_crystal_rs,
                           supercell, N_Ca, N_Si, r_SiOH, r_CaOH, MCL, write_lammps, write_lammps_erica, write_vasp, write_siesta,
                           prefix, unitcell, orthogonal, shift, diferentiate, dpore, saturation, grid, guest_ions)

            jsample += 1

    print("Generation completed in {: 12.6f} s".format(time.time()-t))

    list_properties = np.array(list_properties)    
    plot_XOH_X(list_properties)
    plot_MCL(list_properties)
    plot_distributions(list_properties)
    plot_water(list_properties)

    get_sorted_log(list_properties)



if check:
    check_SiOH_CaOH_MCL(sorted_bricks, widths, size)
    plot_experimental()





if read_structure:

    size, crystal_rs, water_in_crystal_rs, N_Si, N_Ca, r_SiOH, r_CaOH, MCL = read_brick(size_read, brick_code, water_code, pieces, surface_from_bulk)
    
    entries_crystal, entries_bonds, crystal_dict, water_dict = get_full_coordinates( crystal_rs, water_in_crystal_rs, size, pieces, guest_ions,substitute )

    entries_angle = get_angles(crystal_dict, water_dict, size)

    entries_crystal, N_not_ok, itry = check_move_water_hydrogens(entries_crystal)



    unitcell = np.array([ [6.7352,    0.0 ,      0.0],
                            [-4.071295, 6.209521,  0.0],
                           [0.7037701, -6.2095578, 13.9936836] ])

    supercell = np.zeros((3,3))
    for i in range(3):
        supercell[i,:] = unitcell[i,:]*size[i]


    if surface_separation:
        entries_crystal, supercell = transform_surface_separation(entries_crystal, supercell, unitcell, surface_separation)        

    mypath = os.path.abspath(".")
    path = os.path.join(mypath, "output_Y/")

    name = prefix+"_fromManualCode.data"
    name = os.path.join(path, name)
    get_lammps_input(name, entries_crystal, entries_bonds, entries_angle, supercell, write_lammps_erica) 
    name = prefix+"_fromManualCode.log"
    name = os.path.join(path, name)
    get_log(name, size, crystal_rs, water_in_crystal_rs, N_Ca, N_Si, r_SiOH, r_CaOH, MCL )

    name = prefix+"_fromManualCode.vasp"
    name = os.path.join(path, name)
    get_vasp_input(name, entries_crystal, supercell)

    name = prefix+"_fromManualCode.xyz"
    name = os.path.join(path, name)
    get_xyz_input(name, entries_crystal, supercell)

    name = prefix+"_fromManualCode.fdf"
    name = os.path.join(path, name)
    get_siesta_input(name, entries_crystal, supercell)
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 10:12:48 2025

@author: YERAI
"""
import math
import numpy as np
import copy
from mod_write_Y import *
import sys
import random

def distancia(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.linalg.norm(v1 - v2)

def pores_corrected(supercell, n, shift, rel, coord, H = None, coord1 = None):
    coordp=[]
    coordp1=[] 
    if (round(n/2) != n/2 and shift == False) or (round(n/2) == n/2 and shift == True):
        height = supercell[2,2]/2
    else: 
        height = (supercell[2,2]/n)/2
    r = np.zeros((3))
    coordp = []
    coordp1 = []
    if H != None:
        for a in range(len(coord)):
            #print(a)
            r=coord[a][:].copy()
            rH_1=coord1[2*a][:].copy()
            rH_2=coord1[2*a+1][:].copy()
            if height ==0:
                r[:] += rel*(supercell[2,:])
                rH_1[:] += rel*(supercell[2,:])
                rH_2[:] += rel*(supercell[2,:])
            else:
                if r[2]>=height:
                    r[:] += rel*(supercell[2,:])
                    rH_1[:] += rel*(supercell[2,:])
                    rH_2[:] += rel*(supercell[2,:])

            coordp.append(r.copy())
            coordp1.append(rH_1.copy())
            coordp1.append(rH_2.copy())       
        return coordp, coordp1

    else:
        for a in range(len(coord)):
            #print(a)
            r=coord[a][:].copy()
            if height == 0:
                r[:] += rel*(supercell[2,:])
            else:
                if r[2]>=height:
                    r[:] += rel*(supercell[2,:])
            coordp.append(r.copy())
        if coordp == None:
            coordp = []
        return coordp
    

def mallado(supercell, rel, grid, n, shift, entries_crystal, coords_Ca):
    r_ini = []
    #nz = supercell[2,2]/unitcell[2,2]
    if (round(n/2) != n/2 and shift == False) or (round(n/2) == n/2 and shift == True):
        height = supercell[2,2]/2
    else: 
        height = (supercell[2,2]/n)/2
    low_Ca = []
    for i in range(len(coords_Ca)):
        if coords_Ca[i][2]<height:
            low_Ca.append(coords_Ca[i][2])
    maxi_Ca = max(low_Ca)
    maxi_Ca += 3
    rel_dist_2 = maxi_Ca/supercell[2,2]
    r_ini = rel_dist_2*supercell[2,:]
    rel_h=height/supercell[2,2]
    r_ini = rel_h*supercell[2,:]
    r_act = copy.deepcopy(r_ini)
    numx = grid[0]
    numy = grid[1]
    numz = (grid[2]+2)
    x_vec = supercell[0,:]/numx
    y_vec = supercell[1,:]/numy
    z_vec = rel*supercell[2,:]/numz
    coords_filling = []
    filled_atoms = []
    filled_Hw,filled_Ow = [],[]
    cont=len(entries_crystal)
    n = 0 #contador x
    m = 0 #contador y
    j = 1 #contador z
    too_close = False
    printed = False
    while j <= numz-2:
        if too_close:
            break
        r_act = r_ini + m*y_vec + n*x_vec + j*z_vec
        m=0
        while m <= numy-1:
            if too_close:
                break
            n=0
            r_act = r_ini + m*y_vec + n*x_vec + j*z_vec 
            while n <= numx-1:
                if too_close:
                    break
                r_act = r_ini + m*y_vec + n*x_vec + j*z_vec
                coords_filling.append(r_act)
                for w in range(len(coords_Ca)):
                    dist = distancia(r_act, coords_Ca[w])

                    if too_close:
                        break
                    elif dist <= 3: #and j!=0
                        #too_close = True
                        coords_filling.pop()
                        if printed == False:
                            print("Water to close to Ca, reducing amount of water.")
                            printed = True
                        #print(dist, coords_Ca[w], r_act)
                        #for t1 in range(m,-1,-1):
                         #   for t2 in range(n, -1, -1):
                                #coords_filling.pop()
                            #n = numx-1
                        break
                n+=1
            m+=1
        j+=1
    for i in range(len(coords_filling)):
        phi = random.uniform(0, 2*np.pi)
        theta = random.uniform(0, np.pi)
        r = 0.9572
        ang_OO=104.5*np.pi/180
        bit = random.randint(0, 1)
        if bit==1:
            theta_2 = theta + ang_OO
            phi_2 = phi
        else:
            theta_2 = theta + ang_OO
            phi_2 = phi
        r_Ow = [cont+1, 5, -0.82, coords_filling[i][0], coords_filling[i][1], coords_filling[i][2]]
        filled_atoms.append(r_Ow)
        r_Hw = [cont+2, 7, 0.41, coords_filling[i][0] + r*np.sin(theta)*np.cos(phi), coords_filling[i][1] + r*np.sin(theta)*np.sin(phi), coords_filling[i][2]+r*np.cos(theta)]
        
        r_Hw2 = [cont+3, 7, 0.41, coords_filling[i][0] + r*np.sin(theta_2)*np.cos(phi_2), coords_filling[i][1] + r*np.sin(theta_2)*np.sin(phi_2), coords_filling[i][2]+r*np.cos(theta_2)]
        filled_atoms.append(r_Hw)
        filled_atoms.append(r_Hw2)
        cont += 3
    
    return filled_atoms

def substitute_water(filled_atoms, grid):
    dict_water = {14 : "Na", 15 : "Cl"}
    grid_elements=grid[3:]
    N_atoms = []
    substituted_position = []
    erased_hydrogen = []
    final_atoms = []
    new_entries = []
    #print(len(filled_atoms))
    for i in range(int(len(grid_elements)/2)):
        n = float(grid_elements[2*i+1])
        #print(n)
        N_atoms.append(n)
        while (int(N_atoms[i]) <1 ):
            print("Few "+ grid_elements[2*i] + " atoms to reach " + grid_elements[2*i+1] +"%, increasing the amount 1%.")
        N_substituted = 0
        while N_substituted < N_atoms[i]:
            position = random.randint(0,len(filled_atoms)/3)
            match = False
            if filled_atoms[3*position] not in substituted_position:
                substituted_position.append(filled_atoms[3*position])
                erased_hydrogen.append(filled_atoms[3*position+1])
                erased_hydrogen.append(filled_atoms[3*position+2])
                N_substituted += 1
                for t in range(len(list(dict_water.values()))):
                    if list(dict_water.values())[t] == grid_elements[2*i]:
                        specie = list(dict_water.keys())[t]
                        match = True
                if match == False:
                    print("Element not available, change " + grid_elements[2*i])
                    exit
                filled_atoms[3*position][1] = specie  
                filled_atoms[3*position][2] = 0 
                new_entries.append(filled_atoms[3*position])
            
    for entry in filled_atoms:
        if entry not in substituted_position and entry not in erased_hydrogen:
            final_atoms.append(entry)
    for entry in new_entries:
        final_atoms.append(entry)

    return final_atoms
            
            
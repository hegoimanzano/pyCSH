import numpy as np
seed = 23137
size = (2,2,2)   # Minimum (1,1,1)
Ca_Si_ratio = 1.2
W_Si_ratio  = 0.75

prefix = "CaSi"+str(Ca_Si_ratio)

N_samples = 15
make_independent = True

offset_gaussian = False
width_Ca_Si = 0.04
width_SiOH = 0.08
width_CaOH = 0.08

create =True
check = False

orthogonal = True
shift = True
diferentiate = True
dpore = 20


guest_ions = False
substitute = np.array([["Ca1", "Zn", 5, 0.848],["Ca2", "Mn", 5, 0.848]], dtype = object) #sustituted ele, sustitute ele , sustitution %, charge
saturation = True
grid = np.array([4, 5, 6, "Cl", 5, "Na", 5], dtype = object)


write_lammps = True
write_lammps_erica = False
write_vasp = False
write_siesta = False

# The input below allows to read a handmade brick code

read_structure = False

shape_read = (3,3,2)
brick_code = { 
(  0,   0,   0)  :   ['<Lo', 'CU', '<R', '>L', 'CD', 'oMDR', '>R'], 
(  0,   0,   1)  :   ['<L', '<R', 'XD', 'CIU', 'oDL', '>Lo', '>R'], 
(  0,   1,   0)  :   ['<L', 'CU', '<R', 'XU', 'oUL', '>L', '>Ro'], 
(  0,   1,   1)  :   ['<L', '<R', 'XU', 'XD', 'oUL', 'oXU', '>Lo', '>R'], 
(  0,   2,   0)  :   ['<L', 'CU', '<R', 'CII', 'oDR', '>Lo', '>R'], 
(  0,   2,   1)  :   ['<Lo', 'CU', 'oMUL', '<R', 'XD', '>L', '>R'], 
(  1,   0,   0)  :   ['<L', 'SU', 'oMUL', '<R', 'CII', 'XU', 'XD', 'CID', 'CIU', 'oDL', 'oUR', 'oXU', '>L', 'SDo', 'oMDL', '>R'], 
(  1,   0,   1)  :   ['<L', '<R', 'XU', 'oDL', '>Lo', 'CD', '>R'], 
(  1,   1,   0)  :   ['<L', 'CU', '<R', '>L', 'CD', 'oMDR', '>R'], 
(  1,   1,   1)  :   ['<L', 'SUo', 'oMUL', '<R', 'CII', 'XU', 'XD', 'CID', 'CIU', 'oDL', 'oUR', 'oXU', '>L', 'SD', 'oMDL', '>R'], 
(  1,   2,   0)  :   ['<L', 'CU', '<R', 'CII', 'oDR', '>Lo', '>R'], 
(  1,   2,   1)  :   ['<Lo', 'CU', 'oMUL', '<R', 'XU', '>L', '>R'], 
(  2,   0,   0)  :   ['<Lo', '<R', 'XD', 'oUL', 'oXD', '>L', 'CD', '>R'], 
(  2,   0,   1)  :   ['<L', 'SU', 'oMUL', '<R', 'CII', 'XU', 'XD', 'CID', 'CIU', 'oDL', 'oUR', 'oXU', '>L', 'SD', 'oMDL', '>R'], 
(  2,   1,   0)  :   ['<Lo', '<R', 'XD', 'CIU', 'oDL', 'oUR', '>L', '>R'], 
(  2,   1,   1)  :   ['<Lo', '<R', 'XD', '>L', 'CD', 'oMDR', '>R'], 
(  2,   2,   0)  :   ['<L', 'CU', 'oMUL', 'oMUR', '<Ro', 'CID', '>L', '>R'], 
(  2,   2,   1)  :   ['<L', 'CU', '<R', 'CII', 'oDL', '>L', '>Ro'], 
}

water_code = { 
(  0,   0,  0)  :   ['wMDL', 'wXD', 'wUL', 'wIR2', 'wIR'], 
(  0,   0,  1)  :   ['wDR', 'wXD', 'wMDR', 'wMUR', 'w15'], 
(  0,   1,  0)  :   ['wMUR', 'wDR', 'wMUL', 'w15', 'wMDR'], 
(  0,   1,  1)  :   ['w16', 'wIR', 'w15', 'wIR2', 'wIL'], 
(  0,   2,  0)  :   ['wXD', 'wIR', 'w16', 'wMUL', 'wIR2'], 
(  0,   2,  1)  :   ['w14', 'wDR', 'wIL', 'wXD', 'wIR'], 
(  1,   0,  0)  :   ['wXD', 'w16', 'w15', 'wIL'], 
(  1,   0,  1)  :   ['wMDR', 'wIL', 'wIR2', 'wIR'], 
(  1,   1,  0)  :   ['wMUR', 'w15', 'wDR', 'w16'], 
(  1,   1,  1)  :   ['wIR2', 'w14', 'w15', 'wXD'], 
(  1,   2,  0)  :   ['wMDL', 'w16', 'wUL', 'wIR2'], 
(  1,   2,  1)  :   ['wIL', 'wXD', 'wIR', 'wMUR'], 
(  2,   0,  0)  :   ['wMDL', 'w14', 'wMUR', 'w15'], 
(  2,   0,  1)  :   ['w16', 'wIR2', 'w14', 'wIL'], 
(  2,   1,  0)  :   ['wXU', 'wMUL', 'wUL', 'w14'], 
(  2,   1,  1)  :   ['wIL', 'wMDL', 'w16', 'wXD'], 
(  2,   2,  0)  :   ['w16', 'w15', 'wXD', 'wIL'], 
(  2,   2,  1)  :   ['wMUR', 'wIR', 'wIL', 'wXD'], 
}


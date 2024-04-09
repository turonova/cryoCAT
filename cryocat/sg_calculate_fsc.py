import numpy as np
import re
import sys
from matplotlib import pyplot as plt
import cryocat
from cryocat.exceptions import UserInputError

import math

### temporary
import pandas as pd

## load inputs
def load_for_fsc(refA_name,
                refB_name,
                fsc_mask_name=None,
                symmetry='C1'):#write_outputs, pixelsize)
    """Reads 2 reference maps to compare, reads or creates the mask, applies input symmetry and mask.
    
    Parameters
    ----------
    refA_name, refB_name: str or (NP NDARRAY - from cryomap.read doc?!??). Paths to maps to compare.
   TODO fsc_mask_name: str, np.ndarray?!?!??!, default None. Path to mask to be applied. If none, creates a mask of ones.
    symmetry: str. Map symmetry to apply. Default C1.

    Returns
    -------

    Raises
    ------
    UserInputError
        If the input map sizes do not match each other and/or mask, if provided.

    """

    refA = cryocat.cryomap.read(refA_name)
    refB = cryocat.cryomap.read(refB_name)
    ### mask the volumes; in case no mask was provided create an empty mask the size of the input box
    if len(refA[0]) == len(refB[0]):            
        if fsc_mask_name is not None:
            fsc_mask=cryocat.cryomap.read(fsc_mask_name)
            if len(refA[0]) != len(fsc_mask[0]):
                raise UserInputError('Provided mask does not match the size of the map.')
        else:
            print("Creating an empty square mask...")
            fsc_mask=np.ones(cryocat.cryomask.get_correct_format(len(refA),len(refA)))
    else:
        raise UserInputError('Sizes of volumes to compare do not match!')
    mrefA=np.multiply(refA, fsc_mask)
    mrefB=np.multiply(refB, fsc_mask)

    return refA, refB, fsc_mask, mrefA, mrefB, symmetry

def symmterize_volume(vol, symmetry): 

## TODO apply symmetry: based on cryomotl.split_in_asymetric_subunits
## nice example: https://stackoverflow.com/questions/28467446/problems-with-using-re-findall-in-python
    if isinstance(symmetry, str):
        nfold = int(re.findall(r"\d+", symmetry)[-1])
        # if symmetry.lower().startswith("c"):
        #     s_type = 1  # c symmetry
        # elif symmetry.lower().startswith("d"):
        #     s_type = 2  # d symmetry
        # else:
        #         ValueError("Unknown symmetry - currently only c and are supported!")
    elif isinstance(symmetry, (int, float)):
            # s_type = 1  # c symmetry
            nfold = symmetry
    else:
        ValueError("The symmetry has to be specified as a string (starting with C or D) or as a number (float, int)!")
    ## assuming the particle axes correspond to the coordinate system of the box, 
    ## i.e. cyclic rotation is established along the "z" = 3rd dimension of the box
    inplane_step = 360 / nfold
    ## for rotations that are multiplications of 90 degrees
    if symmetry.lower().startswith("c") and nfold == 2:
        symmetrized_vol = np.divide(np.add(vol, np.rot90(vol, k=2, axes=(0, 1))), nfold*np.ones(np.shape(np.asarray(vol))))
    elif symmetry.lower().startswith("c") and nfold == 4:
        symmetrized_vol = np.divide((vol + np.rot90(vol, k=1, axes=(0,1)) + np.rot90(vol, k=2, axes=(0, 1)) + np.rot90(vol, k=3, axes=(0, 1))), nfold*np.ones(np.shape(np.asarray(vol)))) # for i in range(1, nfold))
### TODO will probably crash as the vol will be a seen as tuple

### cryomap rotate - call only with theta 0,0,smth; initate box with 0 and keep adding


    return symmetrized_vol

def calculate_fsc(mrefA, mrefB):
    ### apply FFT to the whole box with no padding over every axis (TODO introduce cutoff); shift 0 to the middle of the box TODO check that it is indeed the case
    ### "The default normalization ("backward") has the direct (forward) transforms unscaled and the inverse (backward) transforms scaled by 1/n"
    mftA = np.fft.fftshift(np.fft.fftn(mrefA))
    mftB = np.fft.fftshift(np.fft.fftn(mrefB))

    ### calculate structure factor arrays
    fsc_denominator = np.multiply(mftA, np.conjugate(mftB))
    fsc_A = np.multiply(mftA, np.conjugate(mftA))  ## = |F1|^2
    fsc_B = np.multiply(mftB, np.conjugate(mftB))  ## = |F2|^2

    ### calculate shells
    ### TODO test for odd and even len numbers - odd numbers should not exist
    # B: this is the distance array - should go most likely to mathutils or somewhere as a standalone function -> I did in mathutils, needs testing first
    shell_grid = np.arange(math.floor(-len(mrefA[0]) / 2), math.ceil(len(mrefA[0]) / 2), 1)
    xv, yv, zv = shell_grid, shell_grid, shell_grid
    shell_space = np.meshgrid(xv, yv, zv, indexing="xy")  ## 'ij' denominates matrix indexing, 'xy' cartesian
    distance_v = np.sqrt(shell_space[0] ** 2 + shell_space[1] ** 2 + shell_space[2] ** 2)

    max_n_shell = math.floor(len(mrefA[0]) / 2)

    ### Precalculate shell masks
    shell_mask = []
    for r in range(0, max_n_shell + 1):
        ### Shells are set to one pixel size
        shell_start = r - 1
        shell_end = r

        ### Generate shell mask
        temp_mask = (distance_v >= shell_start) & (distance_v < shell_end)
        
        ### Write out linearized shell mask
        shell_mask.append(temp_mask.astype(int))

    AB_cc_array = []    # Complex conjugate of A and B
    intA_array = []     # Intenisty of A
    intB_array = []     # Intenisty of B

    full_fsc = [] 

    for r in range(len(shell_mask)):
        AB_cc_array.append(np.sum(shell_mask[r] * fsc_denominator))
        # print(np.sum(shell_mask[r] * fsc_denominator))
        intA_array.append(np.sum(shell_mask[r] * fsc_A))
        # print(np.sum(shell_mask[r] * fsc_A))
        intB_array.append(np.sum(shell_mask[r] * fsc_B))
        # print(np.sum(shell_mask[r] * fsc_B))
        #print("fsc_r", AB_cc_array/np.sqrt(np.multiply(intA_array, intB_array)))
        # print(type(AB_cc_array), type(intA_array), type(intB_array))

        full_fsc.append(np.real(np.divide(AB_cc_array[r], np.sqrt(np.multiply(intA_array[r], intB_array[r])))))
    # print('full_fsc_r', full_fsc_r) ## this one looks better whilst plotting; remember nan is the first value???
    # full_fsc = np.real(AB_cc_array/np.sqrt(np.multiply(intA_array, intB_array)))
    # full_fsc.reshape(17,1)
    # print("full_fsc", full_fsc)
    # full_fsc = np.real(AB_cc_array/math.sqrt(intA_array * intB_array))
    

    return full_fsc

def phase_randomised_reference(mft,
                            fourier_cutoff=None):
    amp, phase = np.absolute(mft), np.angle(mft)
    pr_phase = phase.copy()
    ### calculate fourier_cutoff based on box size
    if fourier_cutoff is None:
        edge = len(mft[0])
        if edge < 75:
            fourier_cutoff = math.floor(edge/5)
        elif edge < 150: 
            fourier_cutoff = math.floor(edge/10)
        else:
            fourier_cutoff = 15
    ### TODO test for fsc len that's close or equal to fourier cutoff


    ### TODO so far gives error:
    ### distance_v = cryocat.mathutils.distance_array(mft)
    shell_grid = np.arange(math.floor(-len(mft[0]) / 2), math.ceil(len(mft[0]) / 2), 1)
    xv, yv, zv = shell_grid, shell_grid, shell_grid
    shell_space = np.meshgrid(xv, yv, zv, indexing="xy")  ## 'ij' denominates matrix indexing, 'xy' cartesian
    distance_v = np.sqrt(shell_space[0] ** 2 + shell_space[1] ** 2 + shell_space[2] ** 2)


    R_fourier_mask = (distance_v >= fourier_cutoff)
    rng = np.random.default_rng()
    randomised_phases_list= rng.permuted((R_fourier_mask * pr_phase).flatten())
    masked_indices = np.where(R_fourier_mask)

    ### Randomize the phases for the masked indices
    pr_phase[masked_indices] = randomised_phases_list[:len(masked_indices[0])]

    ### Compute the randomized phases
    phase_rand = amp * np.exp(1j * pr_phase)

    ### Generate phase-randomised real space maps:
    pr_ref = np.fft.ifftn(np.fft.ifftshift(phase_rand))
    
    return pr_ref

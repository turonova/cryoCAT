# import emfile
# import mrcfile
import numpy as np
# from typing import Optional

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
### TODO should I normalise under mask?
    mrefA=np.multiply(refA, fsc_mask)
    mrefB=np.multiply(refB, fsc_mask)
## TODO APPLY symmetry?
    # B: yes, a function "symmetrize" should be called here, but the function itself should be in cryomap
    return refA, refB, fsc_mask, mrefA, mrefB, symmetry
  

def calculate_fsc(mrefA, mrefB):
    ### apply FFT to the whole box with no padding over every axis (TODO introduce cutoff); shift 0 to the middle of the box
    ### TODO normalisation??? "The default normalization ("backward") has the direct (forward) transforms unscaled and the inverse (backward) transforms scaled by 1/n"
    # B: should not be necessary to normalize here
    mftA = np.fft.fftshift(np.fft.fftn(mrefA))
    mftB = np.fft.fftshift(np.fft.fftn(mrefB))

    ### calculate structure factor arrays
    fsc_denominator = np.multiply(mftA, np.conjugate(mftB))
    fsc_A = np.multiply(mftA, np.conjugate(mftA))  ## = |F1|^2
    fsc_B = np.multiply(mftB, np.conjugate(mftB))  ## = |F2|^2

    ### calculate shells
    ### TODO test for odd and even len numbers
    # B: this is the distance array - should go most likely to mathutils or somewhere as a standalone function -> I did in mathutils, needs testing first
    shell_grid = np.arange(math.floor(-len(mrefA[0]) / 2), math.ceil(len(mrefA[0]) / 2), 1)
    xv, yv, zv = shell_grid, shell_grid, shell_grid
    shell_space = np.meshgrid(xv, yv, zv, indexing="xy")  ## 'ij' denominates matrix indexing, 'xy' cartesian
    distance_v = np.sqrt(shell_space[0] ** 2 + shell_space[1] ** 2 + shell_space[2] ** 2)

    max_n_shell = math.floor(len(mrefA[0]) / 2)

    # Precalculate shell masks
    shell_mask = []
    for r in range(0, max_n_shell + 1):
        # Shells are set to one pixel size
        shell_start = r - 1
        shell_end = r

        # Generate shell mask
        temp_mask = (distance_v >= shell_start) & (distance_v < shell_end)
        
        # Write out linearized shell mask
        shell_mask.append(temp_mask.astype(int))
    ##########shell_mask.append(distance_v.astype(int)) for (distance_v >= shell_edge - 1) and (distance_v < shell_edge) for shell_edge in range(1, max_n_shell+1)

    AB_cc_array = []    # Complex conjugate of A and B
    intA_array = []     # Intenisty of A
    intB_array = []     # Intenisty of B


    full_fsc_r = [] ### DELETE LATER
    full_fsc = np.empty([1,len(shell_mask)])

    import sys
    import numpy
    numpy.set_printoptions(threshold=sys.maxsize)
    for r in range(len(shell_mask)):
        
        print("r", r, "mask",  shell_mask[r][16][16])#, np.where(np.multiply(shell_mask[r],fsc_denominator)))
       
        # print("previous 893869. 893547. 894000. 894962. 893869. 891337.")
        # print(np.multiply(shell_mask[r][16][16], fsc_denominator[16][16]))
        # print(np.where(shell_mask[r][16][16]*fsc_denominator[16][16]))
        # print(np.real(np.sum(shell_mask[r][16][16]*fsc_denominator[16][16])))


        AB_cc_array.append(np.sum(shell_mask[r] * fsc_denominator))
        print(np.sum(shell_mask[r] * fsc_denominator))
        intA_array.append(np.sum(shell_mask[r] * fsc_A))
        print(np.sum(shell_mask[r] * fsc_A))
        intB_array.append(np.sum(shell_mask[r] * fsc_B))
        print(np.sum(shell_mask[r] * fsc_B))
        #print("fsc_r", AB_cc_array/np.sqrt(np.multiply(intA_array, intB_array)))
        print(type(AB_cc_array), type(intA_array), type(intB_array))

        full_fsc_r.append(np.real(AB_cc_array[r]/np.sqrt(np.multiply(intA_array[r], intB_array[r]))))
    print('full_fsc_r', full_fsc_r) ## this one looks better whilst plotting; remember nan is the first value???
    # full_fsc = np.real(AB_cc_array/np.sqrt(np.multiply(intA_array, intB_array)))
    # full_fsc.reshape(17,1)
    print("full_fsc", full_fsc)
    # full_fsc = np.real(AB_cc_array/math.sqrt(intA_array * intB_array))
    return full_fsc, full_fsc_r


def phase_randomised_reference(mft,
                            fourier_cutoff=5):
    amp, phase = np.absolute(mft), np.angle(mft)
    pr_phase = phase.copy()
    
   ### TODO so far gives error
   # ## distance_v = cryocat.mathutils.distance_array(mft)
    shell_grid = np.arange(math.floor(-len(mft[0]) / 2), math.ceil(len(mft[0]) / 2), 1)
    xv, yv, zv = shell_grid, shell_grid, shell_grid
    shell_space = np.meshgrid(xv, yv, zv, indexing="xy")  ## 'ij' denominates matrix indexing, 'xy' cartesian
    distance_v = np.sqrt(shell_space[0] ** 2 + shell_space[1] ** 2 + shell_space[2] ** 2)


    R_fourier_mask = (distance_v >= fourier_cutoff)
    rng = np.random.default_rng()
    randomised_phases_list= rng.permuted((R_fourier_mask * pr_phase).flatten())
    masked_indices = np.where(R_fourier_mask)

    # Randomize the phases for the masked indices
    pr_phase[masked_indices] = randomised_phases_list[:len(masked_indices[0])]

    # Compute the randomized phases
    phase_rand = amp * np.exp(1j * pr_phase)

    ## generate phase-randomised real space maps:
    pr_ref = np.fft.ifftn(np.fft.ifftshift(phase_rand))
    
    return pr_ref

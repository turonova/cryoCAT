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
            print("No FSC mask provided. Creating an empty mask...")
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
        intA_array.append(np.sum(shell_mask[r] * fsc_A))
        intB_array.append(np.sum(shell_mask[r] * fsc_B))
        #print("fsc_r", AB_cc_array/np.sqrt(np.multiply(intA_array, intB_array)))
        full_fsc.append(np.real(np.divide(AB_cc_array[r], np.sqrt(np.multiply(intA_array[r], intB_array[r])))))
    # print('full_fsc_r', full_fsc_r) ## this one looks better whilst plotting; remember nan is the first value???
    # full_fsc = np.real(AB_cc_array/np.sqrt(np.multiply(intA_array, intB_array)))
    # full_fsc.reshape(17,1)
    # print("full_fsc", full_fsc)
    # full_fsc = np.real(AB_cc_array/math.sqrt(intA_array * intB_array))
    return full_fsc

def phase_randomised_reference(mft, fourier_cutoff):
    amp, phase = np.absolute(mft), np.angle(mft)
    pr_phase = phase.copy()

    ### TODO so far gives error: distance_v = cryocat.mathutils.distance_array(mft)
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

def calculate_full_fsc(refA_name,
                refB_name,
                pixelsize_ang=None,
                fsc_mask_name=None,
                symmetry='C1', 
                init_fourier_cutoff=None,
                n_repeats=20,               
                write_outputs=False, ## combine together
                output_folder=None,
                ):
    
    refA, refB, fsc_mask, mrefA, mrefB, symmetry = load_for_fsc(refA_name,
                refB_name,
                fsc_mask_name,
                symmetry)
    
    ## check input px size
    try:
        pixelsize = float(pixelsize_ang)
        print('Pixel size equal', pixelsize, 'Plotting FSC as the function of resolution.')
    except ValueError:
        print('Pixel size not provided. Plotting FSC as the function of shell radius.') # TODO what happens with resolution in the plot?
        pixelsize = 0
    except TypeError:
        print('Pixel size not provided. Plotting FSC as the function of shell radius.') 
        pixelsize = 0

    def substitute_neg_or_nan(fsc_list):
        fsc_list = [1 if math.isnan(x) else x for x in fsc_list]
        fsc_list = [-x if x<0 else x for x in fsc_list]
        return fsc_list
    
    ## calculate FSC
    full_fsc = substitute_neg_or_nan(calculate_fsc(mrefA, mrefB))

    ## calculate fourier_cutoff based on box size
    edge = len(refA[0])
    if init_fourier_cutoff is False:
        print('No Fourier cutoff provided - no randomisation will be executed.') 
        # fourier_cutoff = int(edge/2)

    elif init_fourier_cutoff is None:
        if edge < 25:
            fourier_cutoff = 5
        elif edge < 75:
            fourier_cutoff = math.floor(edge/5)
        elif edge < 150: 
            fourier_cutoff = math.floor(edge/10)
        else:
            fourier_cutoff = 15
        print('Based on the box size, taking', fourier_cutoff, 'as Fourier cutoff value.')
    else:
        fourier_cutoff = init_fourier_cutoff
        print('Fourier cutoff provided and equal', init_fourier_cutoff)
## TODO set as percentage? of not set, do not do randomisation == set to half; without param set print and not randomise

    ## calculate FSC corrected by phase-randomisation normalised in n_repeats
        n_pr_fsc = np.zeros((n_repeats, len(full_fsc)))

        mftA = np.fft.fftshift(np.fft.fftn(mrefA))
        mftB = np.fft.fftshift(np.fft.fftn(mrefB))
        mean_pr_fsc=[] # FIXME do I do it twice???

        for n in range(n_repeats):
            pr_ref_A = phase_randomised_reference(mftA, fourier_cutoff) ## TODO test different fourier cutoff values/box sizes
            pr_ref_B = phase_randomised_reference(mftB, fourier_cutoff)

            ## apply masks
            pr_mref_A = pr_ref_A * fsc_mask
            pr_mref_B = pr_ref_B * fsc_mask

            ## calculate phase-randomised fsc n_repeat times
            rp_fsc_values = calculate_fsc(pr_mref_A, pr_mref_B)
            n_pr_fsc[n,:] = np.divide(np.subtract(full_fsc, rp_fsc_values), np.subtract(1, rp_fsc_values)) ### HERE SIZE [n,:] IS IMPORTANT
            mean_pr_fsc.append(rp_fsc_values)

        np.asarray(mean_pr_fsc)
        #mean_pr_fsc = np.reshape(len(mean_pr_fsc), (int(np.divide(len(mean_pr_fsc), len(rp_fsc_values))), len(rp_fsc_values)))
        mean_pr_fsc = substitute_neg_or_nan(np.mean(mean_pr_fsc, axis=0))

        ## calculate corrected FSC
        corr_fsc = np.mean(n_pr_fsc, axis=0)
        corr_fsc = [1 if math.isnan(x) else x for x in corr_fsc] ## in BT compute_fsc: flip for 1st element if negative
        for i in range(fourier_cutoff+1): 
            corr_fsc[i] = full_fsc[i]

    ## output .csv files with values

    # TODO evaluate FSC flag
    ## generate plot
    fig, ax = plt.subplots() ## TODO understand why fig, ax
    ax.set_title('Fourier Shell Correlation')
    ax.set_yticks([0, 0.143, 0.5, 1])

    # ax.set_xticks(range(0, len(full_fsc))) ## swap for resolution labels
    if pixelsize:
        res_label = [pixelsize*8, pixelsize*4, pixelsize*2, pixelsize]
        ax.set_xticks([len(refA)*pixelsize/i for i in res_label], res_label)
        ax.set_xlabel('Resolution [1/A]') 
    else:
        ax.set_xticks(range(len(refA)+1))
        ax.set_xlabel('Shell radius [px]') 
    # print([len(refA)*pixelsize/i for i in res_label], res_label)
    ax.set_xlim(0, len(full_fsc)-1)
    ## TODO do also 1/A -> do plotting separately
    ## TODO - plot depending on Nyquist

    ax.set_ylim(bottom=(min(full_fsc[1:])-0.05), top=1.01)
    ax.set_ylabel('Fourier Shell Correlation')

    fsc_plot, = plt.plot(full_fsc, color='xkcd:light navy blue')

    plt.plot(mean_pr_fsc, color='xkcd:dull red')
    if init_fourier_cutoff: ### TODO what with none?
        ax.set_ylim(bottom=(min(mean_pr_fsc)-0.01), top=1.01)
    else:
        fsc_copy = full_fsc
        ax.set_ylim(bottom=min(fsc_copy.append(0))-0.01, top=1.01)

    corr_fsc_plot, = plt.plot(corr_fsc, color='black', linewidth=1.5)

    if not fourier_cutoff:
        ax.legend(['Uncorrected FSC', 'Phase-randomised FSC', 'Corrected FSC'])

    ## get values at 0.5, 0.143
    if init_fourier_cutoff == (1 or None):
        xdata = corr_fsc_plot.get_xdata()
        ydata = corr_fsc_plot.get_ydata()
        xnew = np.linspace(0, max(xdata), num=1001) ## TODO!!!  equal spacing instead of num!!!
        ynew = np.interp(xnew, xdata, ydata)
    else:
        xdata = fsc_plot.get_xdata()
        ydata = fsc_plot.get_ydata()
        xnew = np.linspace(0, max(xdata), num=1001) ## TODO!!!  equal spacing instead of num!!!
        ynew = np.interp(xnew, xdata, ydata)

    ## check if the value for 0.5 exists
    if not np.less_equal(ynew, 0.5).any():
        print('FSC does not cross 0.5!')
    else:
        try:
            idx_05 = np.min(np.where(np.isclose(ynew, 0.5, atol=1e-3)))
        except ValueError:
            idx_05 = np.min(np.where(np.isclose(ynew, 0.5, atol=1e-2)))
        finally:
            if not pixelsize:
                print('FSC at 0.5 is', xnew[idx_05])
            print('FSC at 0.5 is', round(len(mrefA)*pixelsize/xnew[idx_05], 3))

    ## check if the value for 0.143 exists
        if not np.less_equal(ynew, 0.143).any():
            print('FSC does not cross 0.143!')
        else:
            try:
                idx_0143 = np.min(np.where(np.isclose(ynew, 0.143, atol=1e-3)))
            except ValueError:
                idx_0143 = np.min(np.where(np.isclose(ynew, 0.143, atol=1e-2)))
            finally:
                if not pixelsize:
                    print('FSC at 0.143 is', xnew[idx_0143])
                print('FSC at 0.143 is', round(len(mrefA)*pixelsize/xnew[idx_0143], 2))

    ## plot additional lines for clarity
    plt.axhline(y=0.5, linestyle='--', color='grey', linewidth=0.75)        
    plt.axhline(y=0.143, linestyle='--', color='grey', linewidth=0.75)
    plt.axhline(y=0,  color='black', linewidth=0.75)

    ## print info about the curve crossing zero
    if not np.less_equal(ynew, 0).any():
        print('FSC does not cross zero!')
    
    # TODO look deeper into cryomap.calculate_flcf flipping for fftshift

    ## TODO reappearing warning: 
    # /Users/makubanska/cryoCAT/cryocat/sg_calculate_fsc.py:136: RuntimeWarning: invalid value encountered in divide
    #   full_fsc.append(np.real(np.divide(AB_cc_array[r], np.sqrt(np.multiply(intA_array[r], intB_array[r])))))

import numpy as np
import pandas as pd
import re
from matplotlib import pyplot as plt
from cryocat.exceptions import UserInputError
import cryocat
import math
import xml.etree.ElementTree as ET

## load inputs
def load_for_fsc(refA_name,
                refB_name,
                fsc_mask_name=None,
                symmetry='C1'):
    """Reads 2 reference maps to compare, reads or creates the mask, applies input symmetry and mask.
    
    Parameters
    ----------
    refA_name, refB_name: str or (NP NDARRAY - from cryomap.read doc?!??). Paths to maps to compare.
    fsc_mask_name: str, np.ndarray TODO test?!?!??!, default None. Path to mask to be applied. If none, creates a mask of ones.
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
    ## mask the volumes; in case no mask was provided create an empty mask the size of the input box
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
        raise ValueError("The symmetry has to be specified as a string (starting with C or D) or as a number (only for C)!")
    ## assuming the particle axes correspond to the coordinate system of the box, 
    ## i.e. cyclic rotation is established along the "z" = 3rd dimension of the box
    inplane_step = 360 / nfold
#     ## for rotations that are multiplications of 90 degrees
#     if symmetry.lower().startswith("c") and nfold == 2:
#         symmetrized_vol = np.divide(np.add(vol, np.rot90(vol, k=2, axes=(0, 1))), nfold*np.ones(np.shape(np.asarray(vol))))
#     elif symmetry.lower().startswith("c") and nfold == 4:
#         symmetrized_vol = np.divide((vol + np.rot90(vol, k=1, axes=(0,1)) + np.rot90(vol, k=2, axes=(0, 1)) + np.rot90(vol, k=3, axes=(0, 1))), nfold*np.ones(np.shape(np.asarray(vol)))) # for i in range(1, nfold))
# ### TODO will probably crash as the vol will be a seen as tuple

### cryomap rotate - call only with theta 0,0,smth; initate box with 0 and keep adding

    return symmetrized_vol

def calculate_fsc(mrefA, mrefB):
    ### "The default normalization ("backward") has the direct (forward) transforms unscaled and the inverse (backward) transforms scaled by 1/n"
    mftA = np.fft.fftshift(np.fft.fftn(mrefA))
    mftB = np.fft.fftshift(np.fft.fftn(mrefB))

    ### calculate structure factor arrays
    fsc_denominator = np.multiply(mftA, np.conjugate(mftB))
    fsc_A = np.multiply(mftA, np.conjugate(mftA))  ## = |F1|^2
    fsc_B = np.multiply(mftB, np.conjugate(mftB))  ## = |F2|^2

    ### calculate shells
    if len(mrefA[0])%2 != 0:
        raise ValueError('The box size should not be an odd number!')
    else:
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
        full_fsc.append(np.real(np.divide(AB_cc_array[r], np.sqrt(np.multiply(intA_array[r], intB_array[r])))))

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
                n_repeats=20,  
                init_fourier_cutoff=None,
                output_table=None,
                output_figure=None,
                ):
    
    refA, refB, fsc_mask, mrefA, mrefB, symmetry = load_for_fsc(refA_name,
                refB_name,
                fsc_mask_name,
                symmetry)
    
    ## check input px size
    if pixelsize_ang != 0:
        try:
            pixelsize = float(pixelsize_ang)
            print(f'Pixel size provided as {pixelsize}. Plotting FSC as the function of resolution.')
        except ValueError:
            print('Pixel size not provided. Plotting FSC as the function of shell radius.')
            pixelsize = 0
        except TypeError:
            print('Pixel size not provided. Plotting FSC as the function of shell radius.') 
            pixelsize = 0
    else:
        print('Pixel size not provided. Plotting FSC as the function of shell radius.') 
        pixelsize = 0

    def substitute_neg_or_nan(fsc_list):
        fsc_list = [1 if math.isnan(x) else x for x in fsc_list]
        fsc_list = [-x if x<0 else x for x in fsc_list]
        return fsc_list
    
    ## calculate FSC
    full_fsc = substitute_neg_or_nan(calculate_fsc(mrefA, mrefB))

    ## generate plot
    fig, ax = plt.subplots()
    ax.set_title('Fourier Shell Correlation')
    ax.set_yticks([0, 0.143, 0.5, 1])

    if pixelsize:
        res_label = [pixelsize * i for i in [8, 4, 2, 1]]
        ax.set_xticks([len(refA)*pixelsize/i for i in res_label], res_label)
        ax.set_xlabel('Resolution [1/A]') 
    else:
        res_label = list(range(1, len(full_fsc)+1))
        ax.set_xticks(range(len(refA)+1))
        ax.set_xlabel('Shell radius [px]') 
    ax.set_xlim(0, len(full_fsc)-1)
    ## TODO do plotting separately??

    ax.set_ylim(bottom=(min(full_fsc[1:])-0.05), top=1.01)
    ax.set_ylabel('Correlation coefficient')

    fsc_plot, = plt.plot(full_fsc, color='xkcd:light navy blue')

    ######
    ## conditional calculation of phase-separated maps and FSC correction
    if n_repeats == 0 or n_repeats == None:
        print('No number of repeats provided - no randomisation will be executed.') 
    else:
        edge = len(refA[0])
    ## calculate fourier_cutoff based on box size        
        if init_fourier_cutoff is None:
            if edge < 100:
                fourier_cutoff = math.floor(edge/10)
            elif edge < 210: 
                fourier_cutoff = math.floor(edge/15) ## TODO test with something HUGE
            else:
                fourier_cutoff = 15
            print('Based on the box size, taking', fourier_cutoff, 'as Fourier cutoff value.')
        elif type(init_fourier_cutoff) == int:
                fourier_cutoff = init_fourier_cutoff
                print(f'Fourier cutoff provided and equal {init_fourier_cutoff}')
        else:
            raise TypeError('The value of Fourier cutoff is neither integer nor None - provide correct input.')
            # TODO should the function still work treating it as the condition with None?

    ## calculate FSC corrected by phase-randomisation normalised in n_repeats
        n_pr_fsc = np.zeros((n_repeats, len(full_fsc)))
        mean_pr_fsc=[] 

        mftA = np.fft.fftshift(np.fft.fftn(mrefA))
        mftB = np.fft.fftshift(np.fft.fftn(mrefB))
        
        for n in range(n_repeats):
            pr_ref_A = phase_randomised_reference(mftA, fourier_cutoff)
            pr_ref_B = phase_randomised_reference(mftB, fourier_cutoff)

            ## apply masks
            pr_mref_A = pr_ref_A * fsc_mask
            pr_mref_B = pr_ref_B * fsc_mask

            ## calculate phase-randomised FSC n_repeat times
            rp_fsc_values = calculate_fsc(pr_mref_A, pr_mref_B) ## phase randomised fsc
            mean_pr_fsc.append(rp_fsc_values) # for plotting phase

            ## phase-normalised FSC per iteration
            n_pr_fsc[n,:] = np.divide(np.subtract(full_fsc, rp_fsc_values), np.subtract(1, rp_fsc_values)) ### HERE SIZE [n,:] IS IMPORTANT

        indiv_phases = np.asarray(mean_pr_fsc)
        mean_pr_fsc = substitute_neg_or_nan(np.mean(mean_pr_fsc, axis=0)) # for plotting

        ## calculate corrected FSC
        corr_fsc = np.mean(n_pr_fsc, axis=0)
        corr_fsc = substitute_neg_or_nan(corr_fsc) ## probably not necessary here anymore
        for i in range(fourier_cutoff+1): 
            corr_fsc[i] = full_fsc[i]
        ## plot mean phase-randomised FSC
        plt.plot(mean_pr_fsc, color='xkcd:dull red')
        corr_fsc_plot, = plt.plot(corr_fsc, color='black', linewidth=1.5)

    ## get mean phase-randomised FSC
    if n_repeats:## test TODO with px 0
        mean_pr_fsc_copy = mean_pr_fsc
        mean_pr_fsc_copy.append(0)
        ax.set_ylim(bottom=(min(mean_pr_fsc_copy)-0.01), top=1.01)
        ax.legend(['Uncorrected FSC', 'Phase-randomised FSC', 'Corrected FSC'])

        ## interpolate to get values at points of significance:
        xdata = corr_fsc_plot.get_xdata()
        ydata = corr_fsc_plot.get_ydata()
        xnew = np.arange(0, max(xdata)+0.001, 0.001)
        ynew = np.interp(xnew, xdata, ydata)

    else:
        ax.legend(['FSC'])
        fsc_copy = list(full_fsc)
        fsc_copy.append(0)
        ax.set_ylim(bottom=min(fsc_copy)-0.01, top=1.01)
        
        ## interpolate to get values at points of significance:
        xdata = fsc_plot.get_xdata()
        ydata = fsc_plot.get_ydata()
        xnew = np.arange(0, max(xdata)+0.001, 0.001)
        ynew = np.interp(xnew, xdata, ydata)
    
    ## plot additional lines for clarity
    plt.axhline(y=0.5, linestyle='--', color='grey', linewidth=0.75)        
    plt.axhline(y=0.143, linestyle='--', color='grey', linewidth=0.75)
    plt.axhline(y=0,  color='black', linewidth=0.75)

    def get_x_for_y(x_range, y_interpolated, y_value):
        ## check if the value exists
        if not np.less_equal(y_interpolated, y_value).any():
            print(f'FSC does not cross {y_value}!')
        else:
            try:
                idx_y_val = np.min(np.where(np.isclose(y_interpolated, y_value, atol=1e-3)))
                if pixelsize:
                    print(f'FSC at {y_value} is', round(len(mrefA)*pixelsize/x_range[idx_y_val], 3))
            except ValueError:
                idx_y_val = np.min(np.where(np.isclose(y_interpolated, y_value, atol=1e-2)))
                if pixelsize:
                    print(f'FSC at {y_value} is', round(len(mrefA)*pixelsize/x_range[idx_y_val], 2))
            finally:
                if not pixelsize:
                    print(f'FSC at {y_value} is {round(x_range[idx_y_val], 3)}.')
        return None
    
    ## get values for 0.5, 0.143
    get_x_for_y(xnew, ynew, 0.5)
    get_x_for_y(xnew, ynew, 0.143)
    ## print info about crossing zero
    if not np.less_equal(ynew, 0).any():
        print('FSC does not cross zero!')

    ## write outputs
    fsc_full_df = np.array(full_fsc)
    if n_repeats:
        corr_fsc_df = np.array(corr_fsc)
        mean_pr_fsc_df = np.array(mean_pr_fsc[1:])
        output_df = pd.DataFrame({'uncorrected FSC':fsc_full_df, 'corrected FSC':corr_fsc_df, 'mean phase-randomised FSC':mean_pr_fsc_df})        
        for n in range(n_repeats):
            output_df[f'phase-randomised FSC #{n}'] = indiv_phases[n,:]
    else:
        output_df = pd.DataFrame({'uncorrected FSC':fsc_full_df})

    ## write outputs as files
    def write_xml(filename, x_values, y_values, title, xaxis):
        title="CryoCAT masked-corrected FSC"
        xaxis="Resolution (1/A)"
        root = ET.Element("fsc", title=title, xaxis=xaxis, yaxis="Correlation Coefficient")
        for x, y in zip(x_values, y_values):
            coordinate = ET.SubElement(root, "coordinate")
            ET.SubElement(coordinate, "x").text = "{:.6f}".format(x)
            ET.SubElement(coordinate, "y").text = "{:.6f}".format(y)
        tree = ET.ElementTree(root)
        tree.write(filename, encoding="utf-8", xml_declaration=True)

    if output_table is not None:
        if output_table.endswith('.csv'):
            output_df.to_csv(output_table, index=False)
        elif output_table.endswith('.xlm'):
            if n_repeats and pixelsize:
                write_xml(output_table, x_values=[len(refA)*pixelsize/i for i in res_label],
                          y_values=corr_fsc,
                          title="CryoCAT phase-corrected FSC",
                          xaxis="Resolution (1/A)")
            elif n_repeats and pixelsize == 0:
                write_xml(output_table, x_values=res_label,
                          y_values=corr_fsc,
                          title="CryoCAT phase-corrected FSC",
                          xaxis="Shell radius [px]")
            elif n_repeats == 0 and pixelsize:
                write_xml(output_table, x_values=[len(refA)*pixelsize/i for i in res_label],
                          y_values=full_fsc,
                          title="CryoCAT raw FSC",
                          xaxis="Resolution (1/A)")
            else: # n_repeats == 0 and pixelsize == 0
                write_xml(output_table, x_values=res_label,
                          y_values=full_fsc,
                          title="CryoCAT raw FSC",
                          xaxis="Shell radius [px]")
        else:
            raise ValueError(f'The output file name {output_table} has to end with .csv or .xml!')

    ## save figure as .png
    if output_figure is not None:
        if output_figure.endswith('.png'):
            plt.savefig(output_figure)
        else:
            raise ValueError(f'The output file name {output_figure} has to end with .png!')

    return output_df, fig

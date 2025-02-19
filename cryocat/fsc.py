import numpy as np
import pandas as pd
import cryocat
from cryocat.exceptions import UserInputError
import math
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt
import re

from scipy import interpolate

## load inputs
def load_for_fsc(refA_name,
                refB_name,
                fsc_mask_name=None,
                # symmetry='C1' TODO
                ):
    """
    Load two reference volumes for FSC analysis, apply mask and symmetrise accordingly.
    
    Parameters
    ----------
    refA_name : str
        File path to the first reference volume.
    refB_name : str
        File path to the second reference volume.
    fsc_mask_name : str, optional
        File path to the FSC mask volume. If not provided, an empty mask will be created.
    symmetry : str, optional
        Symmetry of the volumes. Default is 'C1'.
    
    Returns
    -------
    tuple
        Tuple containing the first volume, the second volume, the FSC mask volume,
        the masked and symmetrised first volume, the masked and symmetrised second volume.
    
    Raises
    ------
    UserInputError
        If the input map sizes do not match each other and/or mask, if provided.

    """

    refA = cryocat.cryomap.read(refA_name)
    refB = cryocat.cryomap.read(refB_name)

    # refA = cryocat.cryomap.symmterize_volume(refA, symmetry=symmetry)
    # refB = cryocat.cryomap.symmterize_volume(refB, symmetry=symmetry)

    ## mask the volumes; in case no mask was provided create an empty mask the size of the input box
    if refA.shape == refB.shape:          
        if fsc_mask_name is not None:
            fsc_mask=cryocat.cryomap.read(fsc_mask_name)
            if refA.shape != fsc_mask.shape:
                raise UserInputError('Provided mask does not match the size of the map.')
        else:
            print("No FSC mask provided. Creating an empty mask...")
            fsc_mask=np.ones(cryocat.cryomask.get_correct_format(len(refA),len(refA)))
    else:
        raise UserInputError('Sizes of volumes to compare do not match!')
    mrefA=np.multiply(refA, fsc_mask)
    mrefB=np.multiply(refB, fsc_mask)

    return refA, refB, fsc_mask, mrefA, mrefB

def calculate_fsc(mrefA, mrefB):
    """
    Calculate Fourier Shell Correlation (FSC) between two 3D volumes.
    
        Parameters
        ----------
        mrefA : ndarray
            (masked) 3D volume A.
        mrefB : ndarray
            (masked) 3D volume B.
    
        Returns
        -------
        ndarray
            Array containing the FSC values for different resolution shells.
    
    """
    ### "The default normalization ("backward") has the direct (forward) transforms unscaled and the inverse (backward) transforms scaled by 1/n"
    mftA = np.fft.fftshift(np.fft.fftn(mrefA))
    mftB = np.fft.fftshift(np.fft.fftn(mrefB))

    ### Calculate structure factor arrays
    fsc_denominator = np.multiply(mftA, np.conjugate(mftB))
    fsc_A = np.multiply(mftA, np.conjugate(mftA))  ## = |F1|^2
    fsc_B = np.multiply(mftB, np.conjugate(mftB))  ## = |F2|^2

    ### Calculate shells
    if len(mrefA[0])%2 != 0:
        raise ValueError('The box size should not be an odd number!')
    else:
        # distance_v = cryocat.geom.distance_array(mrefA) ## TODO TEST IF IT'S WORKING
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
    intA_array = []     # Intensity of A
    intB_array = []     # Intensity of B

    full_fsc = [] 

    for r in range(len(shell_mask)):
        AB_cc_array.append(np.sum(shell_mask[r] * fsc_denominator))
        intA_array.append(np.sum(shell_mask[r] * fsc_A))
        intB_array.append(np.sum(shell_mask[r] * fsc_B))
        denominator = np.sqrt(np.multiply(intA_array[r], intB_array[r]))
        denominator = np.where(denominator == 0, np.nan, denominator)
        full_fsc.append(np.real(np.divide(AB_cc_array[r], denominator)))

    return full_fsc

def phase_randomised_reference(mft, fourier_cutoff):
    """
    This function generates a phase-randomised reference from a given multi-dimensional Fourier transform (mft). 
    
    Parameters
    ----------
    mft : ndarray
        The multi-dimensional Fourier transform from which to generate the phase-randomised reference, or here a FT of a masked volume.
    fourier_cutoff : float
        The cutoff value in the Fourier space beyond which the phases are randomised.
    
    Returns
    -------
    pr_ref : ndarray
        The phase-randomised reference generated from the input Fourier transform.
    
    Notes
    -----
    The function first calculates the amplitude and phase of the input Fourier transform. It then calculates a distance array and creates a mask for values in the distance array that are greater than or equal to the Fourier cutoff. The phases corresponding to these masked indices are then randomised. The function finally computes the phase-randomised reference by applying an inverse Fourier transform to the product of the amplitude and the exponential of the randomised phase.
    """
    amp, phase = np.absolute(mft), np.angle(mft)
    pr_phase = phase.copy()

    ### Calculate distance array
    # distance_v = cryocat.geom.distance_array(mft)
    shell_grid = np.arange(math.floor(-len(mft[0]) / 2), math.ceil(len(mft[0]) / 2), 1)
    xv, yv, zv = shell_grid, shell_grid, shell_grid
    shell_space = np.meshgrid(xv, yv, zv, indexing="xy")  ## 'ij' denominates matrix indexing, 'xy' cartesian
    distance_v = np.sqrt(shell_space[0] ** 2 + shell_space[1] ** 2 + shell_space[2] ** 2)

    ### Randomise phases past the given Fourier cutoff value
    R_fourier_mask = (distance_v >= fourier_cutoff)
    rng = np.random.default_rng()
    randomised_phases_list = rng.permuted((R_fourier_mask * pr_phase).flatten())
    masked_indices = np.where(R_fourier_mask)

    ### Randomize the phases for the masked indices
    pr_phase[masked_indices] = randomised_phases_list[:len(masked_indices[0])]

    ### Compute the randomized phases
    phase_rand = amp * np.exp(1j * pr_phase)

    ### Generate phase-randomised real space maps:
    pr_ref = np.fft.ifftn(np.fft.ifftshift(phase_rand))
    
    return pr_ref

def substitute_neg_or_nan(fsc_list):
    """
    Function to substitute negative numbers and NaN values in a list. 
    
    For the first 10% of the list, if a value is negative, it is replaced with its absolute value. 
    For the entire list, if a value is NaN, it is replaced with 1.
    
    Parameters
    ----------
    fsc_list : list
        List of numbers to be processed. Can include negative numbers and NaN values.
    
    Returns
    -------
    fsc_list : list
        Processed list with no negative numbers in the first 10% and no NaN values.
    """
    
    fsc_list = [1 if math.isnan(x) else x for x in fsc_list]
    fsc_list[0:int(math.floor(0.1*len(fsc_list)))] = [-x if x<0 else x for x in fsc_list[0:int(math.floor(0.1*len(fsc_list)))]]
    return fsc_list

def corrected_fsc(refA_name,
                refB_name,
                pixelsize_angstrom=None,
                fsc_mask_name=None,
                symmetry='C1', 
                n_repeats=20,  
                init_fourier_cutoff=None,
                output_table=None,
                ):
    
    refA, _, fsc_mask, mrefA, mrefB = load_for_fsc(refA_name,
                refB_name,
                fsc_mask_name,
                #symmetry
                )         

    ### Calculate FSC
    full_fsc = calculate_fsc(mrefA, mrefB)
    # print('full_fsc',len(full_fsc))
    full_fsc = substitute_neg_or_nan(full_fsc)
    # print('full_fsc',len(full_fsc)) #9
    ### Check input pixel size                  
    if pixelsize_angstrom is None:
        pixelsize = 1
        print('Pixel size not provided. FSC calculated as the function of shell radius.') 
        x_vals = list(range(1, len(full_fsc)+1)) ## TODO test
    else:           
        pixelsize = float(pixelsize_angstrom)
        print(f'Pixel size provided as {pixelsize}. FSC calculated as the function of resolution.')
        x_vals = [round((len(mrefA))*pixelsize/i, 2) for i in range(1,len(full_fsc)+1)] #[i * pixelsize for i in list(range(len(full_fsc)))]

    def get_x_for_y(x_experimental, y_experimental, y_of_interest):
        if not np.less_equal(y_experimental, y_of_interest).any():
                print(f'FSC does not cross {y_of_interest}!')
        else:
            # Find the closest y to y of interest in existing values
            y_working = y_experimental.copy()
            idx_closest = (np.abs(np.subtract(y_working,y_of_interest))).argmin()
            idx_closest = int(idx_closest)
            # print(idx_closest)
            # y_closest = y_working[idx_closest]
            y_2ndclosest = np.min([y_working[int(idx_closest)-1],y_working[int(idx_closest)+1]])
            idx_2ndclosest = y_working.index(y_2ndclosest)
            #x_of_interest = np.abs(np.divide(y_of_interest*(x_experimental[idx_2ndclosest]-x_experimental[idx_closest]),(y_2ndclosest-y_closest)))
            
            #assuming linearity:
            x_of_interest = np.divide(x_experimental[idx_closest]+x_experimental[int(idx_2ndclosest)],2)
            print(f'FSC at {y_of_interest} is {np.round(x_of_interest,2)}')
        return x_of_interest

    ### Conditional calculation of phase-separated maps and FSC correction
    if n_repeats == None: #TODO different interpolation likely needed
        print('No number of repeats provided - no randomisation will be executed.')    
        xnew = x_vals #list(np.arange(0, len(full_fsc)+0.01, 0.01)) #x_vals #ist(np.arange(0, max(x_vals)+0.001, 0.001) )
        ynew = full_fsc

        ### Get values for 0.5, 0.143
        get_x_for_y(xnew, ynew, 0.5)
        get_x_for_y(xnew, ynew, 0.143)

        ### Write output
        fsc_full_df = np.array(full_fsc)
        output_df = pd.DataFrame({'uncorrected FSC':fsc_full_df})

    else:
        edge = len(refA[0])

        ### If no fourier_cutoff provided, calculate based on the box size (arbitrary)     
        if init_fourier_cutoff is None:
            if edge < 100:
                fourier_cutoff = math.floor(edge/10)
            elif edge < 210: 
                fourier_cutoff = math.floor(edge/15)
            else: #TODO test with sth really large
                fourier_cutoff = 15
            print('Based on the box size, taking', fourier_cutoff, 'as Fourier cutoff value.')
        elif type(init_fourier_cutoff) == int:
                fourier_cutoff = init_fourier_cutoff
                print(f'Fourier cutoff provided and equal {init_fourier_cutoff}')
        else:
            raise TypeError('The value of Fourier cutoff is neither integer nor None - provide correct input.')

        ### Calculate FSC corrected by phase-randomisation normalised by n_repeats
        n_pr_fsc = np.zeros((n_repeats, len(full_fsc)))
        # print('n_pr_fsc',n_pr_fsc.shape) #shape 20,9
        mean_pr_fsc = [] 

        mftA = np.fft.fftshift(np.fft.fftn(mrefA))
        mftB = np.fft.fftshift(np.fft.fftn(mrefB)) 
        
        for n in range(n_repeats):
            pr_ref_A = phase_randomised_reference(mftA, fourier_cutoff)
            pr_ref_B = phase_randomised_reference(mftB, fourier_cutoff)

            ### Apply masks
            pr_mref_A = pr_ref_A * fsc_mask
            pr_mref_B = pr_ref_B * fsc_mask

            ### Calculate phase-randomised FSC n_repeat times
            rp_fsc_values = calculate_fsc(pr_mref_A, pr_mref_B) ## phase randomised fsc
            mean_pr_fsc.append(rp_fsc_values) # for plotting phase

            ### Phase-normalised FSC per iteration
            denominator = np.subtract(1, rp_fsc_values)
            # print('denominator', denominator)
            n_pr_fsc[n,:] = np.where(denominator !=0, np.divide(np.subtract(full_fsc, rp_fsc_values), denominator),np.nan) ### HERE SIZE [n,:] IS IMPORTANT

        indiv_phases = np.asarray(mean_pr_fsc)

        # print('indiv_phases',indiv_phases)
        mean_pr_fsc = substitute_neg_or_nan(np.mean(mean_pr_fsc, axis=0))
        # print('mean_pr_fsc',mean_pr_fsc)
        np.reshape(indiv_phases[1:],(n_repeats-1,len(full_fsc))) ## FIXME

        ## Calculate corrected FSC
        corr_fsc = np.mean(n_pr_fsc, axis=0)
        corr_fsc = substitute_neg_or_nan(corr_fsc) ## probably not necessary here anymore ### FSC HAS TO CROSS 0!!!!
        for i in range(fourier_cutoff+1): 
            corr_fsc[i] = full_fsc[i]

        ### Get interpolated values out of corrected FSC
        xnew = x_vals #[round((len(mrefA))*pixelsize/i, 2) for i in range(1,len(corr_fsc)+1)]#np.arange(0, max(x_vals)+0.001, 0.001) 
        ynew = corr_fsc #np.interp(xnew, x_vals, corr_fsc)
        
        ### Get values for 0.5, 0.143
        get_x_for_y(xnew, ynew, 0.5)
        get_x_for_y(xnew, ynew, 0.143)

        if not np.less_equal(ynew, 0).any():
            print('FSC does not cross zero!')

        ### Write outputs
        corr_fsc_df = np.array(corr_fsc)
        fsc_full_df = np.array(full_fsc)
        mean_pr_fsc_df = np.array(mean_pr_fsc) 
        output_df = pd.DataFrame({'uncorrected FSC':fsc_full_df, 'corrected FSC':corr_fsc_df, 'mean randomised phase':mean_pr_fsc_df})        
        for n in range(n_repeats):
            output_df[f'phase-randomised FSC #{n+1}'] = indiv_phases[n] #[n,:] #FIXME 0 it?

    ### Print info about FSC crossing zero
    if not np.less_equal(ynew, 0).any():
        print('FSC does not cross zero!')

    def write_xml(filename, x_values, y_values, title='CryoCAT FSC', xaxis='Resolution [shell no.]'):
        root = ET.Element("fsc", title=title, xaxis=xaxis, yaxis="Correlation Coefficient")
        for x, y in zip(x_values, y_values):
            coordinate = ET.SubElement(root, "coordinate")
            ET.SubElement(coordinate, "x").text = "{:.6f}".format(x)
            ET.SubElement(coordinate, "y").text = "{:.6f}".format(y)
        tree = ET.ElementTree(root)
        tree.write(filename, encoding="utf-8", xml_declaration=True)
        return 
    
    if output_table != None:    
        if output_table.endswith('.csv'):
            output_df.to_csv(output_table,index=False) 
        elif output_table.endswith('.xml'):
            write_xml(output_table, x_vals, output_df)

    return output_df, pixelsize_angstrom

def plot_fsc(calc_fsc_df, pixelsize_angstrom, output_path=None):
    """
    This function generates a plot of Fourier Shell Correlation (FSC) from a given dataframe. The plot can be saved as a .png file.
    
    Parameters
    ----------
    calc_fsc_df : DataFrame
        The input dataframe containing the FSC data. The dataframe should have columns 'uncorrected FSC', 'corrected FSC', and 'mean randomised phase'.
    pixelsize_angstrom : float
        The pixel size in angstroms. If not provided, the pixel size is assumed to be 1.
    output_path : str, optional
        The path where the output .png file will be saved. If not provided, the plot will not be saved. The output file name has to end with .png.
    
    Returns
    -------
    fig : Figure
        The generated FSC plot.
    
    Raises
    ------
    ValueError
        If the output file name does not end with .png.
    
    Notes
    -----
    This function assumes that the input dataframe is correctly formatted and contains the necessary columns. It does not perform any checks on the input dataframe.
    """
    
    half_box_for_nyquist = len(calc_fsc_df)-1
    # print('half_box_for_nyquist',half_box_for_nyquist)

    fig, ax = plt.subplots()
    ax.set_prop_cycle(color=['xkcd:light navy blue', 'black', 'xkcd:dull red'])

    res_label = [16, 8, 4, 3, 2, 1] # TODO plot only above nyquist

    if pixelsize_angstrom is not None :
        pixelsize = pixelsize_angstrom
        res_label = [round(pixelsize*i, 2) for i in res_label] 
        x_res = [round(2*half_box_for_nyquist*pixelsize/i, 2) for i in res_label] # numerator 1/2 of half box size?
        ax.set_xlabel('Global resolution [A]')
        
    elif pixelsize_angstrom is None:
        pixelsize = 1 
        res_label = [i for i in range(0, len(calc_fsc_df), 5)]
        x_res = res_label.copy()
        ax.set_xlabel('Fourier shell')
    # print('px size set', pixelsize)

    xdata = [i for i in range(0, len(calc_fsc_df))]

    ## Generate plot
    ax.set_title('Fourier Shell Correlation')
    if 'corrected FSC' in calc_fsc_df.columns:
        ax.plot(xdata, calc_fsc_df[['uncorrected FSC', 'corrected FSC', 'mean randomised phase']])
        ax.legend(['Uncorrected FSC', 'Corrected FSC', 'Mean randomised phase'])
    else:
        ax.plot(xdata, calc_fsc_df['uncorrected FSC'])
        ax.legend(['Uncorrected FSC'])
    # print('data',calc_fsc_df[['uncorrected FSC']])

    ax.axhline(y=0.5, linestyle='--', color='grey', linewidth=0.75)        
    ax.axhline(y=0.143, linestyle='--', color='grey', linewidth=0.75)
    ax.axhline(y=0, linestyle='-', color='black', linewidth=0.75)

    ax.set_xticks(x_res, res_label)
    ax.tick_params('x', labelsize=8)
    # ax.set_xlabel('Global resolution [A]')
    ax.set_xlim(0, len(calc_fsc_df)-1)#0.8*len(fsc_data_list)-1) 

    ax.set_yticks([0, 0.143, 0.5, 1])
    # print('calc_fsc_df',np.min(calc_fsc_df))
    ax.set_ylim(bottom=np.nanmin(calc_fsc_df)-0.05, top=1.01) #(np.amin(calc_fsc_df)-0.05)
    # print('ylim',ax.get_ylim())
    ax.set_ylabel('Correlation coefficient')

    # TODO CHECK plotting 1 or 3; col from list + legends

    # plt.plot()


    ## Save figure as .png
    if output_path is not None:
        if output_path.endswith('.png'):
            plt.savefig(output_path)
        else:
            raise ValueError(f'The output file name {output_path} has to end with .png!')
    return fig
    

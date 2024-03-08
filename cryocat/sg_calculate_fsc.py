import emfile
import mrcfile
import numpy as np
from typing import Optional

import cryocat
from cryocat.exceptions import UserInputError

import math

## load inputs
def load_for_fsc(refA_name: str, 
                refB_name: str,
                fsc_mask_name: Optional[str]=None,
                symmetry: Optional[str]='C1'):#write_outputs, pixelsize)
    """Reads 2 reference maps to compare, reads or creates the mask, applies input symmetry and mask.
    
    Parameters
    ----------
    refA_name, refB_name: str or (NP NDARRAY - from cryomap.read doc?!??). Paths to maps to compare.
    fsc_mask_name: str, np.ndarray?!?!??!, default None. Path to mask to be applied. If none, creates a mask of ones.
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
### TODO should the ref be centered/centered under the mask?
    mrefA=np.multiply(refA, fsc_mask)
    mrefB=np.multiply(refB, fsc_mask)
## TODO APPLY symmetry?
    return refA, refB, fsc_mask, mrefA, mrefB, symmetry

#def 

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
    refA = cryocat.cryomap.read(refA_name)
    refB = cryocat.cryomap.read(refB_name)
    if len(refA[0]) == len(refB[0]):            
        if fsc_mask_name is not None:
            fsc_mask=cryocat.cryomap.read(fsc_mask_name)
        else:
            #print("Creating an empty square mask...")
            #fsc_mask=fsc_mask_name
            fsc_mask=np.ones(cryocat.cryomask.get_correct_format(len(refA),len(refA)))
            print(np.linalg.eigvals(fsc_mask))
            ## TODO lengths of inputs - or error -> and test for error
        ##APPLY MASKS
    else:
        raise UserInputError('Sizes of volumes to compare do not match!')
        
    return refA, refB, fsc_mask, symmetry

#def 

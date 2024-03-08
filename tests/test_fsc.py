
import pytest
# import unittest
# import importlib
# importlib.reload(cryocat)
# from cryocat.exceptions import UserInputError
import numpy as np

from cryocat import cryomap
from cryocat import cryomask
from cryocat import sg_calculate_fsc as fsc

# @pytest.mark.parametrize('refA_name', ['/Volumes/msdata/makubans/fsc_python/npc_ref_A_px_1_inv.em'],
#                         'refB_name', ['/Volumes/msdata/makubans/fsc_python/npc_ref_B_px_1_inv.em'],
#                         'fsc_mask_name', ['/Volumes/msdata/makubans/fsc_python/sphere_r10_g2_bx32.em'])


#class UnitTestsMethods(unittest.TestCase):
def test_inputs():
    inputs = fsc.load_for_fsc(refA_name='/Volumes/msdata/makubans/fsc_python/npc_ref_A_px_1_inv.em',
                        refB_name='/Volumes/msdata/makubans/fsc_python/npc_ref_B_px_1_inv.em',
                        fsc_mask_name='/Volumes/msdata/makubans/fsc_python/sphere_r10_g2_bx32.em')
    inputs_cryocat = [cryomap.read('/Volumes/msdata/makubans/fsc_python/npc_ref_A_px_1_inv.em'),
    cryomap.read('/Volumes/msdata/makubans/fsc_python/npc_ref_B_px_1_inv.em'),
    cryomap.read('/Volumes/msdata/makubans/fsc_python/sphere_r10_g2_bx32.em')]
    assert np.allclose(inputs[0:2], inputs_cryocat[0:2])## [0-2] bc load_for_fsc shapes

def test_nomask():
    inputs =  fsc.load_for_fsc(refA_name='/Volumes/msdata/makubans/fsc_python/npc_ref_A_px_1_inv.em',
                        refB_name='/Volumes/msdata/makubans/fsc_python/npc_ref_B_px_1_inv.em',
                        fsc_mask_name=None)
    assert np.array_equal(inputs[2], inputs[2].T)
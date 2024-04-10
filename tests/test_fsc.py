
import pytest
# import unittest
# import importlib
# importlib.reload(cryocat)
# from cryocat.exceptions import UserInputError
import numpy as np

from cryocat import cryomap
from cryocat import cryomask
from cryocat import sg_calculate_fsc as fsc

### trying out parametrization for practice
@pytest.mark.parametrize('refA_name',[('./tests/test_data/npc_ref_A_px_1_inv.em')])
@pytest.mark.parametrize('refB_name', [('./tests/test_data/npc_ref_B_px_1_inv.em')])
@pytest.mark.parametrize('fsc_mask_name', [('./tests/test_data/sphere_r10_g2_bx32.em')])

def test_load_in(refA_name, refB_name, fsc_mask_name):
    inputs = fsc.load_for_fsc(refA_name,
                        refB_name,
                        fsc_mask_name)
    inputs_cryocat = [cryomap.read(refA_name),
    cryomap.read(refB_name),
    cryomap.read(fsc_mask_name)]
    assert np.allclose(inputs[0:2], inputs_cryocat[0:2])## [0-2] bc load_for_fsc output being a tuple (shapes)

def test_none_mask():
    inputs =  fsc.load_for_fsc('./tests/test_data/npc_ref_A_px_1_inv.em',
                        './tests/test_data/npc_ref_B_px_1_inv.em',
                        fsc_mask_name=None)
    assert np.array_equal(inputs[2], inputs[2].T)

def test_apply_none_mask():
    inputs =  fsc.load_for_fsc('./tests/test_data/npc_ref_A_px_1_inv.em',
                        './tests/test_data/npc_ref_B_px_1_inv.em',
                        fsc_mask_name=None)
    assert np.array_equal(inputs[0], inputs[3]) & np.array_equal(inputs[1], inputs[4])

def test_mask_does_something():
    inputs =  fsc.load_for_fsc('./tests/test_data/npc_ref_A_px_1_inv.em',
                        './tests/test_data/npc_ref_B_px_1_inv.em',
                        './tests/test_data/sphere_r10_g2_bx32.em')
    assert np.less_equal(inputs[0], inputs[3]).any() & np.less_equal(inputs[1], inputs[4]).any()

    #pass
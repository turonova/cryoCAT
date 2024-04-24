
import pytest
# import unittest
# import importlib
# importlib.reload(cryocat)
# from cryocat.exceptions import UserInputError
import numpy as np
import csv

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

@pytest.mark.parametrize('refA_name',[('./tests/test_data/npc_ref_A_px_1_inv.em')])
@pytest.mark.parametrize('refB_name', [('./tests/test_data/npc_ref_B_px_1_inv.em')])
@pytest.mark.parametrize('fsc_mask_name', [('./tests/test_data/sphere_r10_g2_bx32.em')])
def test_calculate_fsc(refA_name, refB_name, fsc_mask_name):
    mrefA, mrefB = fsc.load_for_fsc(refA_name, refB_name, fsc_mask_name)
    full_fsc = fsc.calculate_fsc(mrefA, mrefB)
    pass

@pytest.mark.parametrize('refA_name',[('./tests/test_data/npc_ref_A_px_1_inv.em')])
def test_self_fsc(refA_name): ### FIXME why do I have [2:], not just cutting off nan in [0:]
    *_, mrefA, mrefB, _ = fsc.load_for_fsc(refA_name, refA_name)
    ones = [1]*int((len(mrefA)/2)-1)
    full_fsc = fsc.calculate_fsc(mrefA, mrefB)
    np.testing.assert_array_almost_equal(full_fsc[2:], ones, 0.001)

@pytest.mark.parametrize('refA_name',[('./tests/test_data/npc_ref_A_px_1_inv.em')])
@pytest.mark.parametrize('refB_name', [('./tests/test_data/npc_ref_B_px_1_inv.em')])
@pytest.mark.parametrize('fsc_mask_name', [('./tests/test_data/sphere_r10_g2_bx32.em')])
def test_calculate_full_fsc_corr(refA_name, refB_name, fsc_mask_name): ###FIXME with set seed for phase rand.; also tails and the overall vector length are different??
    refA, refB, mask, *_  = fsc.load_for_fsc(refA_name, refB_name, fsc_mask_name)
    corr_fsc = fsc.calculate_full_fsc(refA, refB, 2.682, mask, None, 10, 5)[0] ## [0] - take the resulting df
    print(corr_fsc['corrected FSC'])
    matlab_csv = np.genfromtxt('./tests/test_data/corr_fsc.csv',delimiter=',')
    print(matlab_csv)
    np.testing.assert_allclose(corr_fsc['corrected FSC'][2:], matlab_csv[1:], rtol=0, atol=0.1) ##FIXME why the len different? 

def test_calculate_full_fsc_uncorr(refA_name, refB_name, fsc_mask_name): ##TODO generate matlab uncorrected file
    # loaded = fsc.load_for_fsc(refA_name, refB_name, fsc_mask_name)
    # # auncorr_fsc = fsc.calculate_full_fsc(loaded[2], loaded[3], 2.682, loaded[4], None, 10, 5)[0]
    # matlab_csv = np.genfromtxt('./tests/test_data/corr_fsc.csv',delimiter=',')
    # ### TODO matlab!!!! uncorrected
    # assert np.all(uncorr_fsc['uncorrected FSC'][2:] == matlab_csv[1:])
    assert True


# def test_diff_edges_for_fourier_cutoff:

#     pass
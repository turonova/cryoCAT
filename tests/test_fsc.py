
import pytest
import numpy as np
import pandas as pd
from cryocat import cryomap
from cryocat import cryomask
from cryocat import sg_calculate_fsc as fsc
import matplotcheck.base as mpc


# ### potential parametrization 
# @pytest.mark.parametrize('refA_name',[('./tests/test_data/npc_ref_A_px_1_inv.em')])
# @pytest.mark.parametrize('refB_name', [('./tests/test_data/npc_ref_B_px_1_inv.em')])
# @pytest.mark.parametrize('fsc_mask_name', [('./tests/test_data/sphere_r10_g2_bx32.em')])

@pytest.fixture
def refA_name():
    return './tests/test_data/npc_ref_A_px_1_inv.em'
@pytest.fixture
def refB_name(): 
    return './tests/test_data/npc_ref_B_px_1_inv.em'
@pytest.fixture
def fsc_mask_name():
    return './tests/test_data/sphere_r10_g2_bx32.em'

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

def test_calculate_fsc(refA_name, refB_name, fsc_mask_name):
    mrefA, mrefB = fsc.load_for_fsc(refA_name, refB_name, fsc_mask_name)
    full_fsc = fsc.calculate_fsc(mrefA, mrefB)
    assert True ### FIXME


def test_self_fsc(refA_name): ### FIXME why do I have [2:], not just cutting off nan in [0:]
    *_, mrefA, mrefB, _ = fsc.load_for_fsc(refA_name, refA_name)
    ones = [1]*int((len(mrefA)/2)-1)
    full_fsc = fsc.calculate_fsc(mrefA, mrefB)
    np.testing.assert_array_almost_equal(full_fsc[2:], ones, 0.001)

def test_calculate_full_fsc_corr(refA_name, refB_name, fsc_mask_name): ###FIXME with set seed for phase rand.; also tails and the overall vector length are different??
    refA, refB, mask, *_  = fsc.load_for_fsc(refA_name, refB_name, fsc_mask_name)
    corr_fsc = fsc.calculate_full_fsc(refA, refB, 2.682, mask, 'C1', 10, 5)[0] ## [0] - take the resulting df
    # print(corr_fsc['corrected FSC'])
    matlab_csv = np.genfromtxt('./tests/test_data/corr_fsc.csv',delimiter=',')
    # print(matlab_csv)
    np.testing.assert_allclose(corr_fsc['corrected FSC'][2:], matlab_csv[1:], rtol=0, atol=0.1) ##FIXME why the len different? 

def test_calculate_full_fsc_uncorr(refA_name, refB_name, fsc_mask_name):
    uncorr_fsc = fsc.calculate_full_fsc(refA_name, refB_name, 2.682, fsc_mask_name, 'c1', None, 5)[0]
    # print(uncorr_fsc)
    matlab_csv = [0.999999954578723, 0.995002388859286, 0.981231052592982, 0.954699314186256, 0.917273966043597, 0.921906498703701, 0.836097308533237, 0.736864441953192, 0.629585676265651, 0.524284261860113, 0.387319920161805, 0.203577127473574, 0.177529707480644, 0.252696700444375, 0.206523058965346, 0.160762863339929]
    # print(matlab_csv)
    np.testing.assert_allclose(uncorr_fsc['uncorrected FSC'][1:], matlab_csv, rtol=0, atol=0.1)

def test_inputs_no_randomisation(refA_name, refB_name, fsc_mask_name):
    assert np.all(fsc.calculate_full_fsc(refA_name, refB_name, 2.682, fsc_mask_name, 'c1', None, 5)[0] == fsc.calculate_full_fsc(refA_name, refB_name, 2.682, fsc_mask_name, 'c1', 0, 5)[0])

def test_fcutoff_halfbox_is_norandomisation(refA_name, refB_name, fsc_mask_name):
    df_no_repeats = fsc.calculate_full_fsc(refA_name, refB_name, 2.682, fsc_mask_name, 'c1', None, 5)[0]['uncorrected FSC']
    df_fsc_half_box = fsc.calculate_full_fsc(refA_name, refB_name, 2.682, fsc_mask_name, 'c1', 10, int(len(cryomap.read(refA_name))/2))[0]['corrected FSC']
    pd.testing.assert_series_equal(df_no_repeats, df_fsc_half_box, check_names=False)

def test_error_odd_input_box(odd_volume=cryomask.spherical_mask(21)):
    # odd_volume = cryomask.spherical_mask(21,5)
    with pytest.raises(ValueError) as excinfo:
        mrefA = odd_volume
        mrefB = odd_volume
        fsc.calculate_fsc(mrefA, mrefB)
    assert "odd number" in str(excinfo.value)

def test_xaxislabel_when_no_px_size(refA_name, refB_name, fsc_mask_name):
    shell_radius_plot = fsc.fsc.calculate_full_fsc(refA_name, refB_name, None)

# def test_calculate_fsc():

# def test_phase_randomisatin_w_seed():

# def test_diff_edges_for_fourier_cutoff:

# pass
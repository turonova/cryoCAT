import numpy as np
import pandas as pd
import os
from glob import glob
# import sys
# from pathlib import Path

# path_root = Path(__file__).parents[1]  # given that you are running the script from cryoCAT
# sys.path.append(str(path_root))
# print(sys.path)

from cryocat import pana
import pytest


test_csv_folder = os.path.join('tests', 'test_data', 'pana_data', 'test_template_lists', '*.csv')
test_template_lists = glob(test_csv_folder)

@pytest.mark.parametrize('csv_file', test_template_lists)
def test_create_subtomograms(mocker, csv_file):

    parent_path = './'
    subvolume_sh = np.ones((64, 64, 64))
    angles = [1, 2, 3]

    mocker.patch('cryocat.pana.cut_the_best_subtomo', 
                 return_value=(subvolume_sh, angles))
    mocker.patch('pandas.DataFrame.to_csv') 
   
    df = pana.create_subtomograms_for_tm(csv_file, parent_path)
 
    assert df['Tomo created'].to_list() == [True]*len(df)
    assert all([isinstance(i, str) for i in df['Tomo map'].tolist()]) == True
    assert (df[['Phi', 'Theta', 'Psi']].to_numpy() == np.full((len(df), len(angles)), angles)).all()

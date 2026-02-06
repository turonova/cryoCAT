import numpy as np
import pandas as pd
import pytest

from cryocat import starfileio as sf
from pathlib import Path

test_data = Path(__file__).parent / "test_data"
@pytest.fixture
def relion_optics():
    relion_sf = sf.Starfile(str(test_data / "relion_3.1_optics.star"))
    return relion_sf


# @pytest.mark.parametrize("feature_id", ["score", 5])
def test_relion_optics(relion_optics):
    specifiers = ["data_optics", "data_particles"]
    comments = [["version 30001", "version 30002"], ["version 30001"]]

    assert (
        relion_optics.comments == comments
        and relion_optics.specifiers == specifiers
        and len(relion_optics.frames) == 2
        and relion_optics.frames[0].shape == (1, 7)
        and relion_optics.frames[1].shape == (3, 22)
    )

def test_relion5_star_fix():
    path = str(test_data / "motl_data" / "relion5" / "clean" / "warp2_particles_matching.star")
    #output_path = str(test_data / "motl_data" / "relion5" / "clean" / "warp2_particles_matching_clean.star")
    print(sf.Starfile.fix_relion5_star(path))
    #no exception!

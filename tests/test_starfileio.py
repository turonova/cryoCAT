import numpy as np
import pandas as pd
import pytest

from cryocat import starfileio as sf


@pytest.fixture
def relion_optics():
    relion_sf = sf.Starfile("./tests/test_data/relion_3.1_optics.star")
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

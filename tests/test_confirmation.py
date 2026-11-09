import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/"scripts"))
from confirm_frozen import paired_interval


def test_paired_bootstrap_resamples_clips_after_seed_average():
    # Large opposite seed offsets cancel within every clip. Resampling seed/clip
    # cells independently would invent uncertainty for this constant contrast.
    other = np.ones((2, 20))
    first = other + np.array([[-100], [102]])
    result = paired_interval(first, other, repetitions=100)
    assert result["mean_difference"] == 1
    assert result["clip_bootstrap_percentile_95"] == [1, 1]
    varied = np.arange(40).reshape(2, 20)
    assert paired_interval(varied, other, 100) == paired_interval(varied, other, 100)
    with pytest.raises(ValueError):
        paired_interval(np.ones((3, 20)), other)

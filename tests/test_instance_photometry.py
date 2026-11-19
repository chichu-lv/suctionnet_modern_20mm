"""The source fast path must return exactly the original full-frame PIL pixels."""
import itertools
import numpy as np
import pytest
from suction20.instance_photometry import apply_source_rgb
from suction20.instance_training import _apply_rgb


@pytest.mark.parametrize('region', ['empty','one','sparse','full'])
@pytest.mark.parametrize('has_valid', [True, False])
def test_masked_photometry_matches_full_image(region, has_valid):
    rng = np.random.default_rng(19)
    rgb = rng.uniform(-.1, 1.1, size=(17, 31, 3)).astype(np.float32)
    mask = np.zeros(rgb.shape[:2], dtype=bool)
    if region == 'one':
        mask[-1, 0] = True
    if region == 'sparse':
        mask[1:12, 0:20:2] = True
    if region == 'full':
        mask[:] = True
    rgb[2, 0, 1] = np.nan
    uv = np.array([[0, 2], [0, 16], [15, 4]])
    obs = dict(rgb=rgb, mask=mask, support_uv=uv)
    if has_valid:
        obs['rgb_valid'] = rng.random(mask.shape) > .2
    for order in itertools.permutations(range(3)):
        for gray in (False, True):
            params = ((1.15, .83, 1.1), order, -9 if gray else 7, gray)
            full, fast = _apply_rgb(obs, params), apply_source_rgb(obs, params)
            for key in ['rgb','rgb_valid','support_rgb','support_rgb_valid']:
                assert full[key].dtype == fast[key].dtype
                np.testing.assert_array_equal(full[key], fast[key])

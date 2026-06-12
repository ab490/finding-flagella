import pytest
from pathlib import Path
from predict_model import extract_z, CONF_THRESHOLD, SCATTER_DOT_SCALE


class TestExtractZ:
    def test_standard_filename(self):
        # canonical format written by train_model.py
        assert extract_z(Path("tomo_256717_z0174_y0770_x0514.jpg")) == 174

    def test_z_at_end_before_extension(self):
        assert extract_z(Path("tomo_001_z0099.jpg")) == 99

    def test_slice_prefix(self):
        assert extract_z(Path("slice_0042.jpg")) == 42

    def test_slice_prefix_no_underscore(self):
        assert extract_z(Path("slice0007.jpg")) == 7

    def test_z_prefix_no_leading_underscore(self):
        assert extract_z(Path("z0055_some_stuff.jpg")) == 55

    def test_unparseable_returns_minus_one(self):
        assert extract_z(Path("nozcoord_image.jpg")) == -1

    def test_does_not_pick_y_or_x_coordinate(self):
        # must return 174 (z), not 770 (y) or 514 (x)
        z = extract_z(Path("tomo_001_z0174_y0770_x0514.jpg"))
        assert z == 174

    def test_accepts_string_path(self):
        assert extract_z("tomo_001_z0030_y0100_x0200.jpg") == 30

    def test_leading_zeros_stripped(self):
        assert extract_z(Path("tomo_001_z0001_y0002_x0003.jpg")) == 1


class TestConstants:
    def test_conf_threshold_in_valid_range(self):
        assert 0.0 < CONF_THRESHOLD < 1.0

    def test_scatter_dot_scale_positive(self):
        assert SCATTER_DOT_SCALE > 0

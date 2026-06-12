import numpy as np
import pandas as pd
import pytest
from train_model import normalize_data, split_train_val_test, get_motor_coordinates, BOX_SIZE_PX


def make_labels_df(tomo_ids, motors_per_tomo=1, include_nan=False):
    rows = []
    for i, tid in enumerate(tomo_ids):
        for _ in range(motors_per_tomo):
            rows.append({
                "tomo_id": tid,
                "Number of motors": motors_per_tomo,
                "Motor axis 0": float("nan") if include_nan else 100,
                "Motor axis 1": 200,
                "Motor axis 2": 300,
                "Array shape (axis 0)": 500,
            })
    return pd.DataFrame(rows)


class TestNormalizeData:
    def test_output_range(self):
        arr = np.array([[0, 50, 100, 200, 255]], dtype=np.uint8)
        result = normalize_data(arr)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_dtype_is_uint8(self):
        arr = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        assert normalize_data(arr).dtype == np.uint8

    def test_uniform_array_returns_zeros(self):
        # all-same values -> p2 == p98 -> no contrast to preserve -> return black
        arr = np.full((10, 10), 128, dtype=np.uint8)
        result = normalize_data(arr)
        assert result.dtype == np.uint8
        assert (result == 0).all()

    def test_clips_outliers(self):
        arr = np.array([[0] * 98 + [255, 255]], dtype=np.uint8)
        result = normalize_data(arr)
        # the two extreme values should be clipped into [0,255] without dominating
        assert result.dtype == np.uint8


class TestSplitTrainValTest:
    def test_approximate_ratios(self):
        df = make_labels_df([f"tomo_{i}" for i in range(100)])
        train, val, test = split_train_val_test(df)
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    def test_no_overlap(self):
        df = make_labels_df([f"tomo_{i}" for i in range(30)])
        train, val, test = split_train_val_test(df)
        all_ids = list(train) + list(val) + list(test)
        assert len(all_ids) == len(set(all_ids))

    def test_excludes_zero_motor_tomos(self):
        df = make_labels_df([f"tomo_{i}" for i in range(20)])
        # add tomograms with 0 motors — should be excluded
        no_motor_rows = pd.DataFrame([
            {"tomo_id": "empty_tomo", "Number of motors": 0,
             "Motor axis 0": float("nan"), "Motor axis 1": float("nan"),
             "Motor axis 2": float("nan"), "Array shape (axis 0)": 500}
        ])
        df = pd.concat([df, no_motor_rows], ignore_index=True)
        train, val, test = split_train_val_test(df)
        all_ids = set(train) | set(val) | set(test)
        assert "empty_tomo" not in all_ids

    def test_total_equals_input(self):
        df = make_labels_df([f"tomo_{i}" for i in range(20)])
        train, val, test = split_train_val_test(df)
        assert len(train) + len(val) + len(test) == 20


class TestGetMotorCoordinates:
    def test_returns_correct_structure(self):
        df = make_labels_df(["tomo_1"])
        coords = get_motor_coordinates(df, ["tomo_1"])
        assert len(coords) == 1
        tomo_id, ax0, ax1, ax2, z_depth = coords[0]
        assert tomo_id == "tomo_1"
        assert ax0 == 100
        assert ax1 == 200
        assert ax2 == 300
        assert z_depth == 500

    def test_skips_nan_axis_0(self):
        df = make_labels_df(["tomo_1"], include_nan=True)
        coords = get_motor_coordinates(df, ["tomo_1"])
        assert coords == []

    def test_skips_nan_axis_1(self):
        df = make_labels_df(["tomo_1"])
        df["Motor axis 1"] = float("nan")
        coords = get_motor_coordinates(df, ["tomo_1"])
        assert coords == []

    def test_skips_nan_axis_2(self):
        df = make_labels_df(["tomo_1"])
        df["Motor axis 2"] = float("nan")
        coords = get_motor_coordinates(df, ["tomo_1"])
        assert coords == []

    def test_multiple_motors(self):
        df = make_labels_df(["tomo_1"], motors_per_tomo=3)
        coords = get_motor_coordinates(df, ["tomo_1"])
        assert len(coords) == 3

    def test_unknown_tomo_returns_empty(self):
        df = make_labels_df(["tomo_1"])
        coords = get_motor_coordinates(df, ["tomo_99"])
        assert coords == []


class TestBoxSizePx:
    def test_constant_is_positive(self):
        assert BOX_SIZE_PX > 0

import numpy as np
import pandas as pd
import pytest

from locan.data.locdata_utils import (
    _bump_property_key,
    _check_loc_properties,
    _get_grouped_properties,
    _get_grouped_unique_counts,
    _get_linked_coordinates,
    _get_loc_property_key_per_dimension,
)


@pytest.fixture()
def df_only_coordinates():
    return pd.DataFrame.from_dict(
        {"position_y": np.arange(3), "position_z": np.arange(3)}
    )


@pytest.fixture()
def df_with_zero_uncertainty():
    return pd.DataFrame.from_dict(
        {
            "position_y": np.arange(3),
            "uncertainty": np.arange(3),
            "position_z": np.arange(3),
            "uncertainty_z": np.arange(3),
        }
    )


@pytest.fixture()
def df_with_uncertainty():
    return pd.DataFrame.from_dict(
        {
            "position_y": np.arange(1, 3),
            "uncertainty": np.arange(1, 3),
            "position_z": np.arange(1, 3),
            "uncertainty_z": np.arange(1, 3),
        }
    )


@pytest.fixture()
def df_empty():
    return pd.DataFrame()


@pytest.fixture()
def df_single():
    return pd.DataFrame.from_dict(
        {
            "position_y": [1],
            "position_z": [2],
            "uncertainty_z": [5],
        }
    )


@pytest.fixture()
def df_groups():
    """
    Fixture for returning localizations of three groups:
    rows 0, 2, 4; rows 1, 5 with identical `position_x`; row 3.
    """
    return pd.DataFrame.from_dict(
        {
            "position_x": [0.0, 10, 1, 20, 2, 10],
            "position_y": [0.0, 10, 1, 20, 5, 12],
            "uncertainty_x": [1.0, 2, 2, 3, 1, 2],
            "intensity": [1.0, 4, 2, 6, 3, 5],
            "local_background": [1.0, 2, 1, 6, 4, 4],
            "frame": [3, 7, 1, 9, 2, 8],
        }
    )


def test__get_loc_property_key_per_dimension(df_with_uncertainty):
    results = _get_loc_property_key_per_dimension(
        locdata=df_with_uncertainty, property_key="position"
    )
    assert results == [None, "position_y", "position_z"]

    results = _get_loc_property_key_per_dimension(
        locdata=df_with_uncertainty, property_key="uncertainty"
    )
    assert results == ["uncertainty", "uncertainty", "uncertainty_z"]

    results = _get_loc_property_key_per_dimension(
        locdata=df_with_uncertainty, property_key="not_existing"
    )
    assert results == [None, None, None]


def test__get_linked_coordinates(
    df_only_coordinates,
    df_with_zero_uncertainty,
    df_with_uncertainty,
    df_empty,
    df_single,
    caplog,
):
    results = _get_linked_coordinates(locdata=df_only_coordinates)
    assert results == {
        "position_y": 1.0,
        "uncertainty_y": 0.5773502691896257,
        "position_z": 1.0,
        "uncertainty_z": 0.5773502691896257,
    }

    results = _get_linked_coordinates(locdata=df_with_zero_uncertainty)
    assert all(np.isnan(results[key]) for key in results.keys())
    assert caplog.record_tuples[-1] == (
        "locan.data.locdata_utils",
        30,
        "Zero uncertainties occurred resulting in nan for weighted_mean and weighted_variance.",
    )

    results = _get_linked_coordinates(locdata=df_with_uncertainty)
    assert results == {
        "position_y": 1.2,
        "uncertainty_y": 0.32,
        "position_z": 1.2,
        "uncertainty_z": 0.32,
    }

    results = _get_linked_coordinates(
        locdata=df_only_coordinates, coordinate_keys="position_y"
    )
    assert results == {"position_y": 1.0, "uncertainty_y": 0.5773502691896257}

    results = _get_linked_coordinates(
        locdata=df_with_uncertainty, coordinate_keys="position_y"
    )
    assert results == {
        "position_y": 1.2,
        "uncertainty_y": 0.32,
    }

    results = _get_linked_coordinates(
        locdata=df_only_coordinates, coordinate_keys="position_z"
    )
    assert results == {"position_z": 1.0, "uncertainty_z": 0.5773502691896257}

    results = _get_linked_coordinates(
        locdata=df_with_uncertainty, coordinate_keys="position_z"
    )
    assert results == {
        "position_z": 1.2,
        "uncertainty_z": 0.32,
    }

    results = _get_linked_coordinates(locdata=df_empty)
    assert results == {}

    results = _get_linked_coordinates(locdata=df_single)
    assert results == {
        "position_y": 1,
        "uncertainty_y": 0,
        "position_z": 2,
        "uncertainty_z": 5,
    }


def test__get_grouped_properties(df_groups):
    positions = np.array([0, 2, 4, 1, 5, 3])
    starts = np.array([0, 3, 5])
    results = _get_grouped_properties(
        locdata=df_groups,
        positions=positions,
        starts=starts,
        coordinate_keys=["position_x", "position_y"],
    )
    assert list(results) == [
        "localization_count",
        "position_x",
        "uncertainty_x",
        "position_y",
        "uncertainty_y",
        "intensity",
        "local_background",
        "frame",
        "region_measure_bb",
        "localization_density_bb",
        "subregion_measure_bb",
    ]
    for i, rows in enumerate([[0, 2, 4], [1, 5], [3]]):
        expected = _get_linked_coordinates(locdata=df_groups.iloc[rows])
        assert {key: results[key][i] for key in expected} == pytest.approx(expected)
    assert results["localization_count"].tolist() == [3, 2, 1]
    assert results["intensity"].tolist() == [6, 9, 6]
    assert results["local_background"].tolist() == [2, 3, 6]
    assert results["frame"].tolist() == [1, 7, 9]
    assert results["region_measure_bb"].tolist() == [10, 0, 0]
    assert np.array_equal(
        results["localization_density_bb"], [0.3, np.nan, np.nan], equal_nan=True
    )
    assert np.array_equal(
        results["subregion_measure_bb"], [14, 4, np.nan], equal_nan=True
    )


def test__get_grouped_properties_of_coordinates_only():
    results = _get_grouped_properties(
        locdata=pd.DataFrame({"position_x": [0, 2, 5]}),
        positions=np.arange(3),
        starts=np.array([0, 2]),
        coordinate_keys=["position_x"],
    )
    assert {key: values.tolist() for key, values in results.items()} == {
        "localization_count": [2, 1],
        "position_x": [1, 5],
        "uncertainty_x": [1, 0],
        "region_measure_bb": [2, 0],
        "localization_density_bb": [1, pytest.approx(np.nan, nan_ok=True)],
        "subregion_measure_bb": [4, pytest.approx(np.nan, nan_ok=True)],
    }


def test__get_grouped_properties_sums_small_integers(df_groups):
    df_groups["intensity"] = np.array([200, 1, 200, 1, 200, 1], dtype=np.uint8)
    results = _get_grouped_properties(
        locdata=df_groups,
        positions=np.array([0, 2, 4, 1, 5, 3]),
        starts=np.array([0, 3, 5]),
        coordinate_keys=["position_x", "position_y"],
    )
    assert results["intensity"].tolist() == [600, 2, 1]


@pytest.mark.parametrize(
    "column, values",
    [
        ("intensity", [1, np.nan, 3, 4, 5, 6]),
        ("uncertainty_x", [1, 2, 0, 2, 2, 3]),
        ("frame", pd.array([3, 7, 1, 9, 2, 8], dtype="Int64")),
        ("local_background", list("abcdef")),
    ],
)
def test__get_grouped_properties_for_irregular_values(df_groups, column, values):
    df_groups[column] = values
    results = _get_grouped_properties(
        locdata=df_groups,
        positions=np.array([0, 2, 4, 1, 5, 3]),
        starts=np.array([0, 3, 5]),
        coordinate_keys=["position_x", "position_y"],
    )
    assert results is None


def test__get_grouped_properties_without_groups_or_coordinates(df_groups):
    results = _get_grouped_properties(
        locdata=df_groups,
        positions=np.array([], dtype=np.intp),
        starts=np.array([], dtype=np.intp),
        coordinate_keys=["position_x", "position_y"],
    )
    assert results is None

    results = _get_grouped_properties(
        locdata=df_groups,
        positions=np.array([0, 2, 4, 1, 5, 3]),
        starts=np.array([0, 3, 5]),
        coordinate_keys=[],
    )
    assert results is None


def test__get_grouped_unique_counts():
    results = _get_grouped_unique_counts(
        values=np.array([3, 1, 3, 3, 3, 7]), starts=np.array([0, 3, 5])
    )
    assert results.tolist() == [2, 1, 1]


def test__bump_property_label():
    new_property = _bump_property_key(
        loc_properties=["test", "other_test"],
        loc_property="test",
        extension="_extended",
    )
    assert new_property == "test_extended"

    new_property = _bump_property_key(
        loc_properties=["test", "test_0"],
        loc_property="test",
    )
    assert new_property == "test_0_0"


def test__check_loc_properties(locdata_2d):
    with pytest.raises(ValueError):
        _check_loc_properties(locdata=locdata_2d, loc_properties=["test"])

    result = _check_loc_properties(locdata=locdata_2d, loc_properties="position_x")
    assert result == ["position_x"]

    result = _check_loc_properties(locdata=locdata_2d, loc_properties=None)
    assert result == ["position_x", "position_y"]

    result = _check_loc_properties(
        locdata=locdata_2d, loc_properties=locdata_2d.data.columns
    )
    assert result == locdata_2d.data.columns.tolist()

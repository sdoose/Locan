from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from locan import (
    Bins,
    LocData,
    Rectangle,
    cluster_by_bin,
    cluster_by_radius,
    cluster_dbscan,
    cluster_hdbscan,
    serial_clustering,
)


def test_cluster_hdbscan_2d(locdata_two_cluster_2d):
    noise, clust = cluster_hdbscan(
        locdata_two_cluster_2d, min_cluster_size=3, allow_single_cluster=False
    )
    assert noise.data.empty
    assert len(clust) == 2
    assert all(clust.data.localization_count == [3, 3])
    assert all(clust.references[0].data.cluster_label == 1)
    assert clust.region == locdata_two_cluster_2d.region

    noise, clust = cluster_hdbscan(
        locdata_two_cluster_2d, min_cluster_size=6, allow_single_cluster=True
    )
    assert len(clust) == 1
    assert clust.data.localization_count[0] == 6
    assert all(np.isin(clust.references[0].data.cluster_label, [1, 2]))
    assert clust.region == locdata_two_cluster_2d.region

    noise, clust = cluster_hdbscan(
        locdata_two_cluster_2d,
        loc_properties=["position_x"],
        min_cluster_size=2,
        allow_single_cluster=False,
    )
    assert noise.data.empty
    assert len(clust) == 2
    assert all(clust.data.localization_count == [3, 3])
    assert all(clust.references[0].data.cluster_label == 1)
    assert clust.region == locdata_two_cluster_2d.region


def test_cluster_hdbscan_2d_with_noise(locdata_two_cluster_with_noise_2d):
    noise, clust = cluster_hdbscan(
        locdata_two_cluster_with_noise_2d,
        min_cluster_size=3,
        allow_single_cluster=False,
    )
    assert len(noise) == 1
    assert len(clust) == 2
    assert all(clust.data.localization_count == [3, 3])
    assert all(clust.references[0].data.cluster_label == 1)
    assert noise.region == locdata_two_cluster_with_noise_2d.region
    assert clust.region == locdata_two_cluster_with_noise_2d.region

    noise, clust = cluster_hdbscan(
        locdata_two_cluster_with_noise_2d, min_cluster_size=6, allow_single_cluster=True
    )
    assert len(noise) == 1
    assert len(clust) == 1
    assert clust.data.localization_count[0] == 6
    assert all(np.isin(clust.references[0].data.cluster_label, [1, 2]))
    assert noise.region == locdata_two_cluster_with_noise_2d.region
    assert clust.region == locdata_two_cluster_with_noise_2d.region


def test_cluster_hdbscan_with_shuffled_index(locdata_two_cluster_with_noise_2d):
    locdata = deepcopy(locdata_two_cluster_with_noise_2d)
    new_index = list(locdata.data.index)
    rng = np.random.default_rng()
    rng.shuffle(new_index)
    locdata.data.index = new_index
    noise, clust = cluster_hdbscan(
        locdata, min_cluster_size=3, allow_single_cluster=False
    )
    assert len(noise) == 1
    assert len(clust) == 2
    assert all(clust.data.localization_count == [3, 3])
    assert all(clust.references[0].data.cluster_label == 1)


@pytest.mark.parametrize(
    "fixture_name, expected",
    [("locdata_empty", (0, 0)), ("locdata_single_localization", (1, 0))],
)
def test_cluster_hdbscan_empty_locdata(
    locdata_empty, locdata_single_localization, fixture_name, expected
):
    locdata = eval(fixture_name)
    noise, clust = cluster_hdbscan(
        locdata, min_cluster_size=3, allow_single_cluster=False
    )
    assert len(noise) == expected[0]
    assert len(clust) == expected[1]


def test_cluster_dbscan_2d(locdata_two_cluster_2d):
    noise, clust = cluster_dbscan(locdata_two_cluster_2d, eps=2, min_samples=1)
    assert noise.data.empty
    assert len(clust) == 2
    assert all(clust.data.localization_count == [3, 3])
    assert all(clust.references[0].data.cluster_label == 1)
    assert clust.region == locdata_two_cluster_2d.region

    noise, clust = cluster_dbscan(locdata_two_cluster_2d, eps=20, min_samples=1)
    assert len(clust) == 1
    assert clust.data.localization_count[0] == 6
    assert all(np.isin(clust.references[0].data.cluster_label, [1, 2]))
    assert clust.region == locdata_two_cluster_2d.region

    noise, clust = cluster_dbscan(
        locdata_two_cluster_2d, loc_properties=["position_x"], eps=2, min_samples=1
    )
    assert noise.data.empty
    assert len(clust) == 2
    assert all(clust.data.localization_count == [3, 3])
    assert all(clust.references[0].data.cluster_label == 1)
    assert clust.region == locdata_two_cluster_2d.region

    noise, clust = cluster_dbscan(locdata_two_cluster_2d, eps=0.2, min_samples=10)
    assert len(noise) == len(locdata_two_cluster_2d)
    assert len(clust) == 0
    assert noise.region == locdata_two_cluster_2d.region
    assert clust.region is None


def test_cluster_dbscan_2d_with_noise(locdata_two_cluster_with_noise_2d):
    noise, clust = cluster_dbscan(
        locdata_two_cluster_with_noise_2d, eps=2, min_samples=1
    )
    assert noise.data.empty
    assert len(clust) == 3
    assert all(clust.data.localization_count == [3, 3, 1])
    assert all(clust.references[0].data.cluster_label == 1)
    assert clust.region == locdata_two_cluster_with_noise_2d.region

    noise, clust = cluster_dbscan(
        locdata_two_cluster_with_noise_2d, eps=2, min_samples=2
    )
    assert len(noise) == 1
    assert len(clust) == 2
    assert all(clust.data.localization_count == [3, 3])
    assert all(clust.references[0].data.cluster_label == 1)
    assert noise.region == locdata_two_cluster_with_noise_2d.region
    assert clust.region == locdata_two_cluster_with_noise_2d.region

    noise, clust = cluster_dbscan(
        locdata_two_cluster_with_noise_2d, eps=20, min_samples=2
    )
    assert len(noise) == 1
    assert len(clust) == 1
    assert clust.data.localization_count[0] == 6
    assert all(np.isin(clust.references[0].data.cluster_label, [1, 2]))
    assert noise.region == locdata_two_cluster_with_noise_2d.region
    assert clust.region == locdata_two_cluster_with_noise_2d.region


def test_cluster_dbscan_with_shuffled_index(locdata_two_cluster_with_noise_2d):
    locdata = deepcopy(locdata_two_cluster_with_noise_2d)
    new_index = list(locdata.data.index)
    rng = np.random.default_rng()
    rng.shuffle(new_index)
    locdata.data.index = new_index
    noise, clust = cluster_dbscan(locdata, eps=2, min_samples=2)
    assert len(noise) == 1
    assert len(clust) == 2
    assert all(clust.data.localization_count == [3, 3])
    assert all(clust.references[0].data.cluster_label == 1)  #
    assert noise.region == locdata_two_cluster_with_noise_2d.region
    assert clust.region == locdata_two_cluster_with_noise_2d.region


@pytest.mark.parametrize(
    "fixture_name, expected",
    [("locdata_empty", (0, 0)), ("locdata_single_localization", (1, 0))],
)
def test_cluster_dbscan_empty_locdata(
    locdata_empty, locdata_single_localization, fixture_name, expected
):
    locdata = eval(fixture_name)
    noise, clust = cluster_dbscan(locdata, eps=2, min_samples=2)
    assert len(noise) == expected[0]
    assert len(clust) == expected[1]


@pytest.mark.parametrize(
    "cluster_function, kwargs",
    [
        (cluster_hdbscan, dict(min_cluster_size=3)),
        (cluster_dbscan, dict(eps=2, min_samples=2)),
    ],
)
@pytest.mark.parametrize(
    "region, n_warnings",
    [
        pytest.param(Rectangle((0, 0), 10, 10, 0), 0, id="noise_outside"),
        pytest.param(Rectangle((0, 0), 2, 2, 0), 1, id="centroid_outside"),
    ],
)
def test_cluster_checks_region_for_centroids_only(
    locdata_two_cluster_with_noise_2d,
    caplog,
    cluster_function,
    kwargs,
    region,
    n_warnings,
):
    locdata = deepcopy(locdata_two_cluster_with_noise_2d)
    locdata.region = region
    caplog.clear()
    noise, clust = cluster_function(locdata, **kwargs)
    assert len(noise) == 1
    assert len(clust) == 2
    assert noise.region is region
    assert clust.region is region
    messages = [record.message for record in caplog.records]
    assert messages.count("Not all coordinates are within region.") == n_warnings


# tests serial_clustering


def test_serial_clustering(locdata_two_cluster_with_noise_2d):
    noise, clust = serial_clustering(
        locdata_two_cluster_with_noise_2d,
        cluster_dbscan,
        parameter_lists=dict(eps=[2, 20], min_samples=[2, 5]),
    )
    assert len(noise) == 4
    assert len(clust) == 4


def test_serial_clustering_with_cluster_by_radius(locdata_two_cluster_with_noise_2d):
    noise, clust = serial_clustering(
        locdata_two_cluster_with_noise_2d,
        cluster_by_radius,
        parameter_lists=dict(radius=[1, 4]),
        positions=[[1, 0.8], [5.2, 5.5]],
        return_noise=True,
    )
    assert len(noise) == 2
    assert len(clust) == 2


# tests cluster_by_bin


@pytest.mark.parametrize(
    "fixture_name, expected",
    [
        ("locdata_empty", (np.array([]), 0, np.array([]))),
        ("locdata_single_localization", (np.array([[1, 1]]), 1, np.array([1]))),
    ],
)
def test_cluster_by_bin_empty_locdata(
    locdata_empty, locdata_single_localization, fixture_name, expected
):
    locdata = eval(fixture_name)
    bins, bin_indices, collection, counts = cluster_by_bin(
        locdata,
        loc_properties=["position_x", "position_y"],
        bin_size=2,
        return_counts=True,
    )
    assert bins is None or isinstance(bins, Bins)
    assert np.array_equal(bin_indices, expected[0])
    assert len(bin_indices) == len(collection)
    assert len(collection) == expected[1]
    assert np.array_equal(counts, expected[2])


def test_cluster_by_bin(locdata_2d):
    bins, bin_indices, collection, counts = cluster_by_bin(
        locdata_2d, loc_properties=["position_x", "position_y"], bin_size=5
    )
    # the data-derived range is widened so that the maximum localization is
    # inside the last bin
    assert bins.bin_size == pytest.approx((4, 5))
    assert np.array_equal(bin_indices, [[1, 1]])
    assert len(bin_indices) == len(collection)
    assert counts is None

    bins = Bins.from_n_bins(1, ((1, 5), (1, 6)))
    bins, bin_indices, collection, counts = cluster_by_bin(
        locdata_2d, loc_properties=["position_x", "position_y"], bins=bins
    )
    assert bins.bin_size == (4, 5)
    assert np.array_equal(bin_indices, [[1, 1], [1, 2], [2, 1]])
    assert len(bin_indices) == len(collection)
    assert counts is None

    # test with min_samples
    bins, bin_indices, collection, counts = cluster_by_bin(
        locdata_2d,
        loc_properties=["position_x", "position_y"],
        bin_size=5,
        min_samples=2,
        return_counts=True,
    )
    assert bins.bin_size == pytest.approx((4, 5))
    assert np.array_equal(bin_indices, [[1, 1]])
    assert len(bin_indices) == len(collection)
    assert np.array_equal(counts, [6])


def test_cluster_by_bin_with_non_standard_index(locdata_2d, locdata_non_standard_index):
    _, bin_indices, collection, _ = cluster_by_bin(
        locdata_2d, loc_properties=["position_x", "position_y"], bin_size=5
    )
    _, bin_indices_, collection_, _ = cluster_by_bin(
        locdata_non_standard_index,
        loc_properties=["position_x", "position_y"],
        bin_size=5,
    )
    assert np.array_equal(bin_indices_, bin_indices)
    for reference, reference_ in zip(collection.references, collection_.references):
        assert np.array_equal(reference_.coordinates, reference.coordinates)


# tests cluster_by_radius


def test_cluster_by_radius(locdata_two_cluster_with_noise_2d):
    noise, clust, position_indices = cluster_by_radius(
        locdata_two_cluster_with_noise_2d,
        positions=[[1, 0.8], [50, 50], [5.2, 5.5]],
        radius=1,
    )
    assert noise is None
    assert len(clust) == 2
    assert all(clust.data.localization_count == [3, 3])
    assert np.array_equal(position_indices, [0, 2])
    assert all(clust.references[0].data.cluster_label == 1)
    assert all(clust.references[1].data.cluster_label == 2)
    assert clust.region == locdata_two_cluster_with_noise_2d.region
    assert [item.name for item in clust.meta.history] == ["cluster_by_radius"]


def test_cluster_by_radius_exclusive(locdata_two_cluster_2d):
    positions = [[1, 0.8], [3, 3.2]]
    _, clust, position_indices = cluster_by_radius(
        locdata_two_cluster_2d, positions=positions, radius=4
    )
    assert all(clust.data.localization_count == [3, 6])
    assert np.array_equal(position_indices, [0, 1])

    _, clust, position_indices = cluster_by_radius(
        locdata_two_cluster_2d, positions=positions, radius=4, exclusive=True
    )
    assert all(clust.data.localization_count == [3, 3])
    assert np.array_equal(position_indices, [0, 1])
    assert all(clust.references[0].data.cluster_label == 1)
    assert all(clust.references[1].data.cluster_label == 2)


@pytest.mark.parametrize("exclusive", [False, True])
def test_cluster_by_radius_radius_is_inclusive(locdata_2d, exclusive):
    _, clust, _ = cluster_by_radius(
        locdata_2d, positions=[[1, 3]], radius=2, exclusive=exclusive
    )
    assert clust.references[0].data.index.tolist() == [0, 1, 2]


def test_cluster_by_radius_noise(locdata_two_cluster_with_noise_2d):
    positions = [[1, 0.8], [5.2, 5.5], [100, 100]]
    noise, clust, position_indices = cluster_by_radius(
        locdata_two_cluster_with_noise_2d,
        positions=positions[:2],
        radius=1,
        return_noise=True,
    )
    assert all(noise.data.cluster_label == [-1])
    assert noise.region == locdata_two_cluster_with_noise_2d.region
    assert [item.name for item in noise.meta.history] == ["cluster_by_radius"]
    assert all(clust.data.localization_count == [3, 3])

    noise, clust, position_indices = cluster_by_radius(
        locdata_two_cluster_with_noise_2d,
        positions=positions,
        radius=1,
        return_noise=True,
    )
    assert len(noise) == 0
    assert all(clust.data.localization_count == [3, 3, 1])
    assert np.array_equal(position_indices, [0, 1, 2])

    noise, clust, position_indices = cluster_by_radius(
        locdata_two_cluster_with_noise_2d,
        positions=positions,
        radius=1,
        min_samples=2,
        return_noise=True,
    )
    assert all(noise.data.cluster_label == [-1])
    assert all(clust.data.localization_count == [3, 3])
    assert np.array_equal(position_indices, [0, 1])


@pytest.mark.parametrize(
    "fixture_name, positions, expected",
    [
        ("locdata_empty", [[1, 1]], (0, 0)),
        ("locdata_single_localization", [[1, 1]], (0, 1)),
        ("locdata_single_localization", [[10, 10]], (1, 0)),
        ("locdata_single_localization", [], (1, 0)),
        ("locdata_single_localization", np.empty((0, 0)), (1, 0)),
    ],
)
def test_cluster_by_radius_empty(
    locdata_empty, locdata_single_localization, fixture_name, positions, expected
):
    locdata = eval(fixture_name)
    noise, clust, position_indices = cluster_by_radius(
        locdata, positions=positions, radius=2, return_noise=True
    )
    assert len(noise) == expected[0]
    assert len(clust) == expected[1]
    assert len(position_indices) == len(clust)

    noise, clust, position_indices = cluster_by_radius(
        locdata, positions=positions, radius=2
    )
    assert noise is None
    assert len(clust) == expected[1]


def test_cluster_by_radius_with_non_standard_index(
    locdata_2d, locdata_non_standard_index
):
    positions = [[1, 3], [4, 4]]
    _, collection, position_indices = cluster_by_radius(
        locdata_2d, positions=positions, radius=2.5
    )
    noise, collection_, position_indices_ = cluster_by_radius(
        locdata_non_standard_index, positions=positions, radius=2.5, return_noise=True
    )
    assert np.array_equal(position_indices_, position_indices)
    assert all(collection.data.localization_count == [3, 4])
    for reference, reference_ in zip(collection.references, collection_.references):
        assert np.array_equal(reference_.coordinates, reference.coordinates)
    assert noise.data.index.tolist() == []


def test_cluster_by_radius_keeps_order():
    rng = np.random.default_rng(seed=1)
    locdata = LocData.from_coordinates(rng.uniform(0, 100, size=(1000, 2)))
    _, collection, _ = cluster_by_radius(
        locdata, positions=rng.uniform(0, 100, size=(5, 2)), radius=30
    )
    assert len(collection) == 5
    for reference in collection.references:
        assert reference.data.index.is_monotonic_increasing


@pytest.mark.parametrize(
    "fixture_name, loc_properties, positions, radius, expected",
    [
        ("locdata_1d", None, [[1.5]], 0.5, [0, 1, 2]),
        ("locdata_3d", None, [[1, 3, 2]], 2, [1]),
        ("locdata_3d", ["position_x", "position_y"], [[1, 3]], 2, [0, 1, 2]),
    ],
)
def test_cluster_by_radius_loc_properties(
    locdata_1d, locdata_3d, fixture_name, loc_properties, positions, radius, expected
):
    locdata = eval(fixture_name)
    _, collection, _ = cluster_by_radius(
        locdata, positions=positions, radius=radius, loc_properties=loc_properties
    )
    assert collection.references[0].data.index.tolist() == expected


@pytest.mark.parametrize("exclusive", [False, True])
def test_cluster_by_radius_reduce(locdata_two_cluster_with_noise_2d, exclusive):
    kwargs = dict(positions=[[1, 0.8], [3, 3.2]], radius=4, exclusive=exclusive)
    _, expected, _ = cluster_by_radius(locdata_two_cluster_with_noise_2d, **kwargs)
    _, collection, position_indices = cluster_by_radius(
        locdata_two_cluster_with_noise_2d, reduce=True, **kwargs
    )
    assert collection.references is None
    pd.testing.assert_frame_equal(collection.data, expected.data)
    assert np.array_equal(position_indices, [0, 1])
    assert collection.region == locdata_two_cluster_with_noise_2d.region
    assert [item.name for item in collection.meta.history] == ["cluster_by_radius"]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(positions=[1, 2]),
        dict(positions=[[1, 2, 3]]),
        dict(positions=[[1, 2]], min_samples=0),
    ],
)
def test_cluster_by_radius_raises(locdata_2d, kwargs):
    with pytest.raises(ValueError):
        cluster_by_radius(locdata_2d, radius=2, **kwargs)

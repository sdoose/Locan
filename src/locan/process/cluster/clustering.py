"""

Methods for clustering localization data in LocData objects.

"""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from copy import copy
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.spatial import KDTree

# pyrefly: ignore [missing-module-attribute]
from sklearn.cluster import DBSCAN, HDBSCAN

from locan import config
from locan.data.locdata import LocData
from locan.data.locdata_utils import _check_loc_properties
from locan.process.aggregate import _accumulate_2d, _data_bin_range
from locan.process.bins import Bins

if TYPE_CHECKING:
    import boost_histogram as bh

__all__: list[str] = [
    "cluster_hdbscan",
    "cluster_dbscan",
    "cluster_by_bin",
    "cluster_by_radius",
]

logger = logging.getLogger(__name__)


def cluster_hdbscan(
    locdata: LocData,
    min_cluster_size: int = 5,
    loc_properties: list[str] | None = None,
    allow_single_cluster: bool = False,
    **kwargs: Any,
) -> tuple[LocData, LocData]:
    """
    Cluster localizations in locdata using the hdbscan clustering algorithm.

    Parameters
    ----------
    locdata
        Localization data on which to perform the manipulation.
    loc_properties
        The LocData properties to be used for clustering.
        If None, `locdata.coordinates` will be used.
    min_cluster_size
        Minimumm cluster size in HDBSCAN algorithm (default: 5)
    allow_single_cluster
        If True, return single cluster (default: False)
    kwargs
        Other parameters passed to `hdbscan.HDBSCAN`.

    Returns
    -------
    tuple[LocData, LocData]
        A tuple with noise and cluster.
        The first LocData object is a selection of all localizations that are
        defined as noise, in other words all localizations that are not part
        of any cluster.
        The second LocData object is a LocData instance assembling all
        generated selections (i.e. localization cluster).

    Note
    ----
    In locdata 0.20.0 the original hdbscan implementation was replaced by the
    scikit-learn implementation.
    The new implementation should yield identical results if the min_samples
    parameter is increased by one. See notes at
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html
    """
    parameter = locals()

    if len(locdata) == 0:
        locdata_noise = LocData()
        collection = LocData()

    elif len(locdata) < min_cluster_size:
        locdata_noise = copy(locdata)
        collection = LocData()

    else:
        if loc_properties is None:
            fit_data = locdata.coordinates
        else:
            fit_data = locdata.data[loc_properties].to_numpy()

        labels = HDBSCAN(
            min_cluster_size=min_cluster_size,
            allow_single_cluster=allow_single_cluster,
            **kwargs,
        ).fit_predict(fit_data)

        grouped = locdata.data.groupby(labels)
        locdata_index_labels = [
            locdata.data.index[idxs] for idxs in grouped.indices.values()
        ]
        selections = [
            LocData.from_selection(locdata=locdata, indices=idxs)
            for idxs in locdata_index_labels
        ]

        try:
            grouped.get_group(-1)
            locdata_noise = selections[0]
            collection = LocData.from_collection(selections[1:])
        except KeyError:
            locdata_noise = LocData()
            collection = LocData.from_collection(selections)

    # set regions
    if locdata_noise:
        locdata_noise.region = locdata.region
    if collection:
        collection.region = locdata.region

    # metadata
    if locdata_noise:
        del locdata_noise.meta.history[:]
        locdata_noise.meta.history.add(
            name=sys._getframe().f_code.co_name, parameter=str(parameter)
        )
    del collection.meta.history[:]
    collection.meta.history.add(
        name=sys._getframe().f_code.co_name, parameter=str(parameter)
    )

    return locdata_noise, collection


def cluster_dbscan(
    locdata: LocData,
    eps: float = 20,
    min_samples: int = 5,
    loc_properties: list[str] | None = None,
    **kwargs: Any,
) -> tuple[LocData, LocData]:
    """
    Cluster localizations in locdata using the dbscan clustering algorithm as
    implemented in sklearn.

    Parameters
    ----------
    locdata
        specifying the localization data on which to perform the manipulation.
    eps
        The maximum distance between two samples for them to be considered as
        in the same neighborhood.
    min_samples
        The number of samples in a neighborhood for a point to be considered
        as a core point.
        This includes the point itself.
    loc_properties
        The LocData properties to be used for clustering. If None,
        `locdata.coordinates` will be used.
    kwargs
        Other parameters passed to `sklearn.cluster.DBSCAN`.

    Returns
    -------
    tuple[LocData, LocData]
        A tuple with noise and cluster.
        The first LocData object is a selection of all localizations that are
        defined as noise, in other words all localizations that are not part
        of any cluster.
        The second LocData object is a LocData instance assembling all
        generated selections (i.e. localization cluster).
    """
    parameter = locals()

    if len(locdata) == 0:
        locdata_noise = LocData()
        collection = LocData()

    else:
        if loc_properties is None:
            fit_data = locdata.coordinates
        else:
            fit_data = locdata.data[loc_properties].to_numpy()

        labels = DBSCAN(
            eps=eps, min_samples=min_samples, n_jobs=config.n_cpus, **kwargs
        ).fit_predict(fit_data)

        grouped = locdata.data.groupby(labels)
        locdata_index_labels = [
            locdata.data.index[idxs] for idxs in grouped.indices.values()
        ]
        selections = [
            LocData.from_selection(locdata=locdata, indices=idxs)
            for idxs in locdata_index_labels
        ]

        try:
            grouped.get_group(-1)
            locdata_noise = selections[0]
            collection = LocData.from_collection(selections[1:])
        except KeyError:
            try:
                # if there is just one localization declared as noise, then the
                # previous lookup throws a Key Error.
                grouped.get_group((-1,))
                locdata_noise = selections[0]
                collection = LocData.from_collection(selections[1:])
            except KeyError:
                locdata_noise = LocData()
                collection = LocData.from_collection(selections)

    # set regions
    if locdata_noise:
        locdata_noise.region = locdata.region
    if collection:
        collection.region = locdata.region

    # metadata
    if locdata_noise:
        del locdata_noise.meta.history[:]
        locdata_noise.meta.history.add(
            name=sys._getframe().f_code.co_name, parameter=str(parameter)
        )
    del collection.meta.history[:]
    collection.meta.history.add(
        name=sys._getframe().f_code.co_name, parameter=str(parameter)
    )

    return locdata_noise, collection


def cluster_by_bin(
    locdata: LocData,
    loc_properties: list[str] | None = None,
    min_samples: int = 1,
    bins: Bins | bh.axis.Axis | bh.axis.AxesTuple | None = None,
    n_bins: int | Sequence[int] | None = None,
    bin_size: float | Sequence[float] | Sequence[Sequence[float]] | None = None,
    bin_edges: Sequence[float] | Sequence[Sequence[float]] | None = None,
    bin_range: (
        tuple[float, float] | Sequence[float] | Sequence[Sequence[float]] | str | None
    ) = None,
    return_counts: bool = False,
) -> tuple[Bins | None, npt.NDArray[np.int64], LocData, npt.NDArray[np.int64] | None]:
    """
    Cluster localizations in locdata by binning all localizations with regard
    to `loc_properties` and collecting all localizations per bin as cluster.

    Parameters
    ----------
    locdata
        Localization data.
    loc_properties
        Localization properties to be grouped into bins.
        If None The coordinate_values of locdata are used.
    min_samples
        The minimum number of samples per bin to be considered as cluster.
    bins
        The bin specification as defined in :class:`Bins`
    bin_edges
        Array of bin edges with shape (n_bin_edges,)
        or (dimension, n_bin_edges) for all or each dimension.
    n_bins
        The number of bins for all or each dimension.
        5 yields 5 bins in all dimensions.
        (2, 5) yields 2 bins for one dimension and 5 for the other dimension.
    bin_size
        The size of bins in units of locdata coordinate units for all or each
        dimension.
        5 would describe bin_size of 5 for all bins in all dimensions.
        (2, 5) yields bins of size 2 for one dimension and 5 for the other
        dimension.
        To specify arbitrary sequence of `bin_sizes` use `bin_edges` instead.
    bin_range : tuple[float, ...] | tuple[tuple[float, float], ...] | Literal['zero'] | None
        The data bin_range to be taken into consideration for all or each
        dimension.
        ((min_x, max_x), (min_y, max_y), ...) bin_range for each coordinate;
        for None (min, max) bin_range are determined from data;
        for 'zero' (0, max) bin_range with max determined from data.
    return_counts
        If true, n_elements per bin are returned.

    Returns
    -------
    tuple[Bins | None, npt.NDArray[np.int64], LocData, npt.NDArray[np.int64] | None]
        Tuple with bins, bin_indices,
        collection of all generated selections (i.e. localization clusters),
        and counts per bin.
    """
    parameter = locals()

    if len(locdata) == 0:
        bins = None
        bin_indices = np.array([])
        collection = LocData()
        counts = np.array([]) if return_counts else None
        return bins, bin_indices, collection, counts

    loc_properties = _check_loc_properties(locdata, loc_properties)
    data = locdata.data[loc_properties].values

    if bins is not None:
        bins = Bins.from_any(bins=bins)
        if any([item is not None for item in [n_bins, bin_size, bin_edges, bin_range]]):
            logger.warning("bins are used - all other bin specifications are ignored.")
    else:
        if (bin_range is None or isinstance(bin_range, str)) and bin_edges is None:
            if bin_range is not None and bin_range not in ("zero", "link"):
                raise ValueError(
                    "`bin_range` must be a range, 'zero', 'link' or None, "
                    f"got {bin_range!r}."
                )
            special = cast(Literal["zero", "link"] | None, bin_range)
            bin_range_ = _data_bin_range(locdata, loc_properties, special=special)
        elif isinstance(bin_range, str):
            raise ValueError(
                "`bin_range` must be None when `bin_edges` is given, "
                f"got {bin_range!r}."
            )
        else:
            bin_range_ = bin_range
        bins = Bins.from_any(
            n_bins=n_bins,
            bin_size=bin_size,
            bin_edges=bin_edges,
            bin_range=bin_range_,
        )
    if bins.dimension != len(loc_properties):
        raise ValueError(
            f"The dimension of bins ({bins.dimension}) does not match the number "
            f"of `loc_properties` ({len(loc_properties)})."
        )
    bins = bins.with_labels(loc_properties)

    if bins.dimension == 2:
        bin_indices, data_indices, _, counts = _accumulate_2d(
            data, bins=bins, return_counts=True
        )
        assert counts is not None  # type narrowing # noqa: S101
    else:
        raise NotImplementedError("Only implemented for dimension 2.")

    if min_samples > 1:
        mask = counts >= min_samples
        counts = counts[mask]
        bin_indices = bin_indices[mask]
        data_indices = [
            data_indices_e
            for data_indices_e, mask_e in zip(data_indices, mask)
            if mask_e
        ]

    # data_indices are positions, from_selection takes index labels
    index = locdata.data.index
    selections = [
        LocData.from_selection(locdata=locdata, indices=index[idxs])
        for idxs in data_indices
    ]
    collection = LocData.from_collection(selections)

    # set regions
    if collection:
        collection.region = locdata.region

    # metadata
    del collection.meta.history[:]
    collection.meta.history.add(
        name=sys._getframe().f_code.co_name, parameter=str(parameter)
    )

    if not return_counts:
        counts = None

    return bins, bin_indices, collection, counts


def cluster_by_radius(
    locdata: LocData,
    positions: npt.ArrayLike,
    radius: float,
    min_samples: int = 1,
    loc_properties: list[str] | None = None,
    *,
    exclusive: bool = False,
    reduce: bool = False,
    return_noise: bool = False,
) -> tuple[LocData | None, LocData, npt.NDArray[np.int64]]:
    """
    Cluster localizations in locdata by collecting all localizations within
    `radius` of each position as cluster.

    A localization at a distance equal to `radius` is within radius.
    A localization within radius of more than one position is part of each
    of these clusters unless `exclusive` is true.

    Parameters
    ----------
    locdata
        Localization data.
    positions
        Cluster centers with shape (n_positions, n_loc_properties),
        e.g. the `coordinates` of another LocData object.
    radius
        The maximum distance between a position and a localization for the
        localization to be part of the cluster.
    min_samples
        The minimum number of localizations within radius of a position to
        be considered as cluster.
    loc_properties
        The LocData properties to be used for clustering.
        If None, `locdata.coordinates` will be used.
    exclusive
        If true, each localization is part of the cluster of its nearest
        position only. For equal distances one of the positions is taken.
    reduce
        Return the reduced LocData object or keep references alive.
        The reduced object carries the properties of all clusters but no
        references to a selection of localizations for each cluster;
        setting it up is faster for a large number of clusters.
    return_noise
        If true, noise is returned.

    Returns
    -------
    tuple[LocData | None, LocData, npt.NDArray[np.int64]]
        A tuple with noise, cluster and position_indices.
        The first element is a selection of all localizations that are
        defined as noise, in other words all localizations that are not part
        of any cluster. It is None if `return_noise` is false.
        The second element is a LocData instance assembling all generated
        selections (i.e. localization cluster).
        The third element holds for each cluster the index of the
        corresponding element in `positions`.

    Raises
    ------
    ValueError
        If `min_samples` is smaller than one or if the shape of `positions`
        does not fit the number of `loc_properties`.

    Examples
    --------
    >>> import locan as lc
    >>> locdata = lc.LocData.from_coordinates(
    ...     [(0, 0), (1, 0), (10, 10), (50, 50)]
    ... )
    >>> noise, collection, position_indices = lc.cluster_by_radius(
    ...     locdata, positions=[(0, 0), (30, 30), (10, 10)], radius=2
    ... )
    >>> collection.data.localization_count.tolist()
    [2, 1]
    >>> position_indices.tolist()
    [0, 2]
    """
    # converted before locals() is taken: the history records the abbreviated
    # string of an array instead of the full string of a long list
    positions = np.asarray(positions, dtype=np.float64)
    parameter = locals()

    if min_samples < 1:
        raise ValueError(f"`min_samples` must be at least 1, got {min_samples}.")

    if len(locdata) == 0:
        return (
            LocData() if return_noise else None,
            LocData(),
            np.array([], dtype=np.int64),
        )

    loc_properties = _check_loc_properties(locdata, loc_properties)
    points = locdata.data[loc_properties].to_numpy()

    # labels and data_positions refer to rows in positions and points
    if positions.size == 0:
        n_positions = 0
        labels = np.array([], dtype=np.int64)
        data_positions = np.array([], dtype=np.int64)
    elif positions.ndim != 2 or positions.shape[1] != len(loc_properties):
        raise ValueError(
            f"`positions` must have shape (n_positions, {len(loc_properties)}), "
            f"got {positions.shape}."
        )
    else:
        n_positions = len(positions)
        tree = KDTree(positions)
        if exclusive:
            _, nearest = tree.query(
                points,
                k=1,
                # the bound of KDTree.query is exclusive, radius is inclusive
                distance_upper_bound=np.nextafter(radius, np.inf),
            )
            data_positions = np.flatnonzero(nearest < n_positions)
            labels = nearest[data_positions]
        else:
            other_tree = KDTree(points)
            neighbors = tree.query_ball_tree(other=other_tree, r=radius)
            n_neighbors = np.fromiter(
                (len(neighbors_) for neighbors_ in neighbors),
                dtype=np.int64,
                count=n_positions,
            )
            labels = np.repeat(np.arange(n_positions), n_neighbors)
            data_positions = np.fromiter(
                chain.from_iterable(neighbors),
                dtype=np.int64,
                count=n_neighbors.sum(),
            )
            # query_ball_tree returns the neighbors of a position in
            # arbitrary order; labels are sorted already
            data_positions = data_positions[np.lexsort((data_positions, labels))]

    counts = np.bincount(labels, minlength=n_positions)
    position_indices = np.flatnonzero(counts >= min_samples)
    is_kept = counts[labels] >= min_samples

    # data_positions are positions, selections take index labels
    index = locdata.data.index
    collection = LocData._from_groups(
        locdata=locdata,
        labels=pd.Series(labels[is_kept], index=index[data_positions[is_kept]]),
        reduce=reduce,
    )

    locdata_noise = None
    if return_noise:
        is_noise = np.ones(len(index), dtype=bool)
        is_noise[data_positions[is_kept]] = False
        if is_noise.any():
            locdata_noise = LocData.from_selection(
                locdata=locdata, indices=index[is_noise]
            )
        else:
            locdata_noise = LocData()

    # set regions
    if collection:
        collection.region = locdata.region

    # metadata
    if locdata_noise:
        del locdata_noise.meta.history[:]
        locdata_noise.meta.history.add(
            name=sys._getframe().f_code.co_name, parameter=str(parameter)
        )
    del collection.meta.history[:]
    collection.meta.history.add(
        name=sys._getframe().f_code.co_name, parameter=str(parameter)
    )

    return locdata_noise, collection, position_indices

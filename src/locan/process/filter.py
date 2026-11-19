"""

Filter localization data.

This module provides functions for filtering LocData objects.
The functions take LocData as input and compute new LocData objects.

"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors

from locan import cluster_by_bin, config
from locan.constants import HullType
from locan.data.images import Image
from locan.data.locdata import LocData
from locan.data.metadata_utils import _modify_meta
from locan.data.regions.region import Interval, Region, Region2D, RoiRegion
from locan.data.validation import _check_loc_properties
from locan.locan_types import RandomGeneratorSeed

if TYPE_CHECKING:
    from locan.data import Image

__all__: list[str] = [
    "Selector",
    "filter_condition",
    "select_by_condition",
    "select_by_region",
    "select_by_image_mask",
    "exclude_sparse_points",
    "random_subset",
    "thin_by_distance",
    "localizations_in_cluster_regions",
]

# Tuning of _thin_points for speed and memory; the result does not depend on it.
# Number of points from which the mean number of neighbors is estimated.
_N_SAMPLES: int = 4096
# Mean number of neighbors up to which all close pairs are computed at once.
_MAX_MEAN_NEIGHBORS: int = 10
# Number of points visited per batch if neighbors are queried in batches.
_N_VISITS: int = 512
# Number of neighbors per batch above which a batch is shortened.
_MAX_NEIGHBORS: int = 2**20


class Selector:
    """
    Define selection interval for a single localization property.

    Parameters
    ----------
    loc_property
        Localization property
    activate
        Indicator to apply the selection or not
    lower_bound
        min fo selection interval
    upper_bound
        max of selection interval

    Attributes
    ----------
    loc_property : str
        Localization property
    activate : bool
        Indicator to apply the selection or not
    lower_bound : int | float
        min fo selection interval
    upper_bound: int | float
        max of selection interval
    interval : Interval
        Class with interval specifications
    condition : str
        specification turned into condition string
    """

    def __init__(
        self,
        loc_property: str,
        activate: bool,
        lower_bound: int | float,
        upper_bound: int | float,
    ) -> None:
        self.loc_property = loc_property
        self.activate = activate
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self.interval = Interval(lower_bound=lower_bound, upper_bound=upper_bound)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"loc_property='{self.loc_property}', "
            f"activate={self.activate}, "
            f"lower_bound={self.lower_bound}, "
            f"upper_bound={self.upper_bound}"
            f")"
        )

    @property
    def lower_bound(self) -> int | float:
        return self._lower_bound

    @lower_bound.setter
    def lower_bound(self, value: int | float) -> None:
        self._lower_bound = value
        self.interval = Interval(lower_bound=value, upper_bound=self.upper_bound)

    @property
    def upper_bound(self) -> int | float:
        return self._upper_bound

    @upper_bound.setter
    def upper_bound(self, value: int | float) -> None:
        self._upper_bound = value
        self.interval = Interval(lower_bound=self.lower_bound, upper_bound=value)

    @property
    def condition(self) -> str:
        if self.activate is True:
            condition = f"{self.lower_bound} < {self.loc_property} < {self.upper_bound}"
        else:
            condition = ""
        return condition


def filter_condition(selectors: Iterable[Selector]) -> str:
    """
    Get a condition string from selection specifications.

    Parameters
    ----------
    selectors
        Specifications for loc_property selections

    Returns
    -------
    str
    """
    iterable = (selector.condition for selector in selectors if selector.activate)
    condition = " and ".join(iterable)
    return condition


def select_by_condition(locdata: LocData, condition: str) -> LocData:
    """
    Select by specifying conditions on data properties.

    Parameters
    ----------
    locdata
        Specifying the localization data from which to select.
    condition
        Conditions as input in select method.
        More precise: query specifications to be used with pandas query.

    Returns
    -------
    LocData
        A new instance of LocData referring to the specified dataset.
    """
    local_parameter = locals()

    # select
    new_indices = locdata.data.query(condition).index.values.tolist()

    # instantiate
    new_locdata = LocData.from_selection(locdata=locdata, indices=new_indices)  # type: ignore[arg-type]

    # update metadata
    meta_ = _modify_meta(
        locdata,
        new_locdata,
        function_name=sys._getframe().f_code.co_name,
        parameter=local_parameter,
        meta=None,
    )
    new_locdata.meta = meta_

    return new_locdata


def select_by_region(
    locdata: LocData,
    region: Region,
    loc_properties: list[str] | None = None,
    reduce: bool = True,
) -> LocData:
    """
    Select localizations from `locdata` that are within `region` and return
    a new LocData object.
    Selection is with respect to loc_properties or to localization coordinates
    that correspond to region dimension.

    Parameters
    ----------
    locdata
        Localization data that is tested for being inside the region.
    region
        Tested region
    loc_properties
        Localization properties to be tested.
    reduce
        Return the reduced LocData object or keep references alive.

    Returns
    -------
    LocData
        A new instance of LocData with all localizations within region of
        interest.

    Note
    ----
    Points on boundary of regions are considered outside if region is a
    shapely object, but inside if region is a
    matplotlib (which is the case for RoiRegion) object.
    """
    local_parameter = locals()

    if loc_properties is None:
        loc_properties_ = locdata.coordinate_keys[0 : region.dimension]
    else:
        loc_properties_ = loc_properties

    points = locdata.data[list(loc_properties_)].values

    if isinstance(region, Region2D | RoiRegion):
        indices_inside = region.contains(points)
        locdata_indices_to_keep = locdata.data.index[indices_inside]
        new_locdata = LocData.from_selection(
            locdata=locdata, indices=locdata_indices_to_keep
        )
        new_locdata.set_region(region, check=False)
    else:
        raise NotImplementedError("Only Region2D has been implemented.")

    # finish
    if reduce:
        new_locdata.reduce()

    # update metadata
    meta_ = _modify_meta(
        locdata,
        new_locdata,
        function_name=sys._getframe().f_code.co_name,
        parameter=local_parameter,
        meta=None,
    )
    new_locdata.meta = meta_

    return new_locdata


def select_by_image_mask(
    locdata: LocData,
    image: Image,
    loc_properties: list[str] | None = None,
) -> LocData:
    """
    Select by masking using an image.
    Each pixel is interpreted as True if different from 0.

    Parameters
    ----------
    locdata
        specifying the localization data from which to select.
    image
        Image serving as mask.
    loc_properties
        Localization properties to be filtered.

    Returns
    -------
    LocData
        a new instance of LocData referring to the selected dataset.
    """
    local_parameter = locals()
    loc_properties = _check_loc_properties(locdata, loc_properties)
    assert loc_properties is not None  # type narrowing # noqa: S101
    if image.ndim != len(loc_properties):
        raise TypeError(
            "Dimension of image must correspond to the length of loc_properties"
        )

    bins, bin_indices, collection, counts = cluster_by_bin(
        locdata=locdata, loc_properties=loc_properties, bins=image.bins
    )
    mask = [tuple(item) for item in np.transpose(np.nonzero(image.data)).tolist()]  # type: ignore
    items = [
        collection.references[i]  # type: ignore
        for i, indices in enumerate(bin_indices)
        if tuple(indices - 1) in mask  # type: ignore
    ]
    if items:
        new_index = np.concatenate([item.data.index for item in items])
        new_locdata = LocData.concat(items)
        new_locdata.data.index = new_index
    else:
        new_locdata = LocData()

    # update metadata
    meta_ = _modify_meta(
        locdata,
        new_locdata,
        function_name=sys._getframe().f_code.co_name,
        parameter=local_parameter,
        meta=None,
    )
    new_locdata.meta = meta_

    return new_locdata


def exclude_sparse_points(
    locdata: LocData,
    other_locdata: LocData | None = None,
    radius: float = 50,
    min_samples: int = 5,
) -> LocData:
    """
    Exclude localizations by thresholding a local density.

    A subset of localizations, that exhibit a small local density of
    localizations from locdata or alternatively from other_locdata,
    is identified as noise and excluded.
    Noise is identified by using a nearest-neighbor search
    (:class:`sklearn.neighbors.NearestNeighbors`) to find all
    localizations within a circle (sphere) of the given `radius`.
    If the number of localizations is below the
    threshold value `min_samples`, the localization is considered to be noise.

    The method identifies the same noise points as done by the clustering
    algorithm DBSCAN [1]_.

    Parameters
    ----------
    locdata
        Specifying the localization data from which to exclude localization
        data.
    other_locdata
        Specifying the localization data on which to compute local density.
    radius
        Radius of a circle or sphere in which neighbors are identified
        (equivalent to epsilon in DBSCAN).
    min_samples
        The minimum number of samples in the neighborhood that need to be
        found for each localization to not be
        identified as noise (equivalent to minPoints in DBSCAN).

    Returns
    -------
    LocData
        All localizations except those identified as sparse (noise) points.

    References
    ----------
    .. [1] Martin Ester, Hans-Peter Kriegel, Jörg Sander, Xiaowei Xu,
       A density-based algorithm for discovering clusters in large spatial
       databases with noise.
       In: Evangelos Simoudis, Jiawei Han, Usama M. Fayyad (Hrsg.):
       Proceedings of the Second International Conference
       on Knowledge Discovery and Data Mining (KDD-96). AAAI Press, 1996,
       S. 226-231, ISBN 1-57735-004-9.
    """
    local_parameter = locals()

    if other_locdata is None:
        nn = NearestNeighbors(metric="euclidean", n_jobs=config.n_cpus).fit(
            locdata.coordinates
        )
        neighbor_points_list = nn.radius_neighbors(radius=radius, return_distance=False)
        # if points is not provided the query point is not considered its own neighbor.
    else:
        nn = NearestNeighbors(metric="euclidean", n_jobs=config.n_cpus).fit(
            other_locdata.coordinates
        )
        neighbor_points_list = nn.radius_neighbors(
            locdata.coordinates, radius=radius, return_distance=False
        )

    indices_to_keep = [len(pts) >= min_samples for pts in neighbor_points_list]
    locdata_indices_to_keep = locdata.data.index[indices_to_keep]
    new_locdata = LocData.from_selection(locdata, locdata_indices_to_keep)

    # update metadata
    meta_ = _modify_meta(
        locdata,
        new_locdata,
        function_name=sys._getframe().f_code.co_name,
        parameter=local_parameter,
        meta=None,
    )
    new_locdata.meta = meta_

    return new_locdata


def random_subset(
    locdata: LocData,
    n_points: int,
    replace: bool = True,
    seed: RandomGeneratorSeed = None,
) -> LocData:
    """
    Take a random subset of localizations.

    Parameters
    ----------
    locdata
        Specifying the localization data from which to select localization data.
    n_points
        Number of localizations to randomly choose from locdata.
    replace
        Indicate if sampling is with or without replacement
    seed
        Random number generation seed

    Returns
    -------
    LocData
        A new instance of LocData carrying the subset of localizations.
    """
    local_parameter = locals()

    if not locdata:
        return locdata

    rng = np.random.default_rng(seed)

    indices = rng.choice(locdata.data.index, size=n_points, replace=replace)
    new_locdata = LocData.from_selection(locdata, indices)

    # update metadata
    meta_ = _modify_meta(
        locdata,
        new_locdata,
        function_name=sys._getframe().f_code.co_name,
        parameter=local_parameter,
        meta=None,
    )
    new_locdata = LocData.from_selection(locdata, indices, meta=meta_)

    return new_locdata


def _thin_points(
    points: npt.NDArray[np.float64],
    min_distance: float,
    order: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """
    Visit points in the given order, keep each visited point that has not
    been removed and remove all points within `min_distance` of it.

    Neighbors are only needed for the points that are kept.
    If points have few neighbors, all close pairs are computed in one query.
    Otherwise most points are removed before they are visited and neighbors
    are queried in batches for the points that are left.

    Parameters
    ----------
    points
        Coordinates with shape (n_points, n_dimensions).
    min_distance
        The distance up to which neighbors of a kept point are removed.
    order
        Positions of points in the sequence in which they are visited.

    Returns
    -------
    npt.NDArray[np.int64]
        Positions of the kept points in the sequence given by `order`.
    """
    # from here on, position i refers to the point that is visited at step i
    points = points[order]
    n_points = len(points)
    if n_points == 0:
        return order

    tree = KDTree(points)
    is_removed = np.zeros(n_points, dtype=bool)

    sample = points[:: max(1, n_points // _N_SAMPLES)]
    n_neighbors = tree.query_ball_point(sample, r=min_distance, return_length=True)
    # each count includes the point itself
    if n_neighbors.mean() - 1 <= _MAX_MEAN_NEIGHBORS:
        # pairs are (first, second) with first < second
        pairs = tree.query_pairs(r=min_distance, output_type="ndarray")
        first, second = pairs[:, 0], pairs[:, 1]
        second = second[np.argsort(first, kind="stable")]
        counts = np.bincount(first, minlength=n_points)
        stops = np.cumsum(counts)
        visits = np.flatnonzero(counts)
        for visit, start, stop in zip(
            visits.tolist(),
            (stops - counts)[visits].tolist(),
            stops[visits].tolist(),
            strict=True,
        ):
            if not is_removed[visit]:
                is_removed[second[start:stop]] = True
    else:
        start = 0
        while start < n_points:
            visits = start + np.flatnonzero(~is_removed[start : start + _N_VISITS])
            start += _N_VISITS
            if visits.size == 0:
                continue
            # A list of neighbors per visited point is held in memory:
            # the batch is shortened if a few counts indicate large lists.
            n_neighbors = tree.query_ball_point(
                points[visits[:: max(1, visits.size // 32)]],
                r=min_distance,
                return_length=True,
            )
            n_visits = max(1, int(_MAX_NEIGHBORS / n_neighbors.mean()))
            if n_visits < visits.size:
                start = int(visits[n_visits])
                visits = visits[:n_visits]
            neighbors = tree.query_ball_point(
                points[visits], r=min_distance, return_sorted=False
            )
            for visit, neighbors_ in zip(visits.tolist(), neighbors, strict=True):
                if not is_removed[visit]:
                    is_removed[neighbors_] = True
                    # neighbors_ contains the visited point itself
                    is_removed[visit] = False

    return order[~is_removed]


def thin_by_distance(
    locdata: LocData,
    min_distance: float,
    order: npt.ArrayLike | None = None,
    loc_properties: list[str] | None = None,
    seed: RandomGeneratorSeed = None,
) -> LocData:
    """
    Thin localizations such that all remaining localizations are farther
    apart than `min_distance`.

    Localizations are visited one after the other in the given `order`.
    A visited localization is kept unless it has been excluded before,
    and all localizations within `min_distance` of a kept localization
    are excluded.
    A localization at a distance equal to `min_distance` is excluded.
    Each excluded localization is thus within `min_distance` of a remaining
    one.

    Parameters
    ----------
    locdata
        Specifying the localization data from which to select localization
        data.
    min_distance
        The distance that all pairs of remaining localizations exceed.
    order
        Positions (row numbers, not index labels) of the localizations in
        the sequence in which they are visited.
        Localizations that come first take precedence.
        Localizations that are not listed are excluded.
        If None, localizations are visited in random order.
    loc_properties
        The LocData properties to be used as coordinates.
        If None, `locdata.coordinates` will be used.
    seed
        Random number generation seed for the random order.

    Returns
    -------
    LocData
        A new instance of LocData with the remaining localizations in the
        sequence of `locdata`.

    Raises
    ------
    ValueError
        If `min_distance` is negative.
    TypeError
        If `order` is not a one-dimensional sequence of integers.

    Examples
    --------
    >>> import locan as lc
    >>> locdata = lc.LocData.from_coordinates(
    ...     [(0, 0), (1, 0), (2, 0), (10, 10)]
    ... )
    >>> thinned = lc.thin_by_distance(
    ...     locdata, min_distance=1, order=[0, 1, 2, 3]
    ... )
    >>> thinned.data.index.tolist()
    [0, 2, 3]
    >>> thinned = lc.thin_by_distance(
    ...     locdata, min_distance=1, order=[1, 0, 2, 3]
    ... )
    >>> thinned.data.index.tolist()
    [1, 3]
    """
    # converted before locals() is taken: the history records the abbreviated
    # string of an array instead of the full string of a long list
    if order is not None:
        order = np.asarray(order)
    local_parameter = locals()

    if min_distance < 0:
        raise ValueError(f"`min_distance` must not be negative, got {min_distance}.")

    if len(locdata) == 0:
        return locdata

    loc_properties = _check_loc_properties(locdata, loc_properties)
    points = locdata.data[loc_properties].to_numpy()

    if order is None:
        order_ = np.random.default_rng(seed).permutation(len(points))
    elif order.ndim != 1 or (order.size > 0 and order.dtype.kind not in "iu"):
        # a boolean mask or floats would be taken as positions silently
        raise TypeError(
            "`order` must be a one-dimensional sequence of integer positions."
        )
    else:
        order_ = order.astype(np.int64)

    positions = _thin_points(points=points, min_distance=min_distance, order=order_)
    new_locdata = LocData.from_selection(
        locdata=locdata, indices=locdata.data.index[np.sort(positions)]
    )

    # update metadata
    meta_ = _modify_meta(
        locdata,
        new_locdata,
        function_name=sys._getframe().f_code.co_name,
        parameter=local_parameter,
        meta=None,
    )
    new_locdata.meta = meta_

    return new_locdata


def localizations_in_cluster_regions(
    locdata: LocData,
    collection: LocData | list[LocData],
    hull_type: HullType | str = HullType.CONVEX_HULL,
) -> LocData:
    """
    Identify localizations from `locdata` within the regions of all
    `collection` elements.

    Parameters
    ----------
    locdata
        Localization data that is tested for being inside the region
    collection
        A set of Locdata objects collected in a collection or list.
    hull_type
        The hull type for each LocData object that is used to define the
        region.

    Returns
    --------
    LocData
        A collection of LocData objects with all elements of locdata
        contained by the region.
    """
    locdatas = []
    if isinstance(hull_type, str):
        hull_type = HullType[hull_type.upper()].value
    else:
        hull_type = hull_type.value

    if isinstance(collection, LocData):
        if isinstance(collection.references, list):  # this case covers pure collections
            for ref in collection.references:
                cregion = getattr(ref, hull_type).region
                locdata_selection = select_by_region(locdata=locdata, region=cregion)
                locdatas.append(locdata_selection)
        else:  # this case covers selections of collections
            for index in collection.indices:  # type: ignore
                cregion = getattr(
                    collection.references.references[index], hull_type  # type: ignore
                ).region
                locdata_selection = select_by_region(locdata=locdata, region=cregion)
                locdatas.append(locdata_selection)
    else:  # this case covers list of LocData objects
        for ref in collection:
            cregion = getattr(ref, hull_type).region
            locdata_selection = select_by_region(locdata=locdata, region=cregion)
            locdatas.append(locdata_selection)

    new_collection = LocData.from_collection(locdatas)

    return new_collection

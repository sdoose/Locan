"""

Utility functions for working with locdata.

"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from locan.utils.statistics import weighted_mean_variance

if TYPE_CHECKING:
    from locan.data.locdata import LocData  # noqa F401
    from locan.locan_types import DataFrame  # noqa F401

__all__: list[str] = []

logger = logging.getLogger(__name__)


def _get_loc_property_key_per_dimension(
    locdata: pd.DataFrame | pd.Series[Any], property_key: str
) -> list[str | None]:
    """
    Get tuple with property_key as available in each dimension.
    For the x-coordinates precedence is given as:
    `position_x` or `position` or None.
    None if the key is not available for a certain dimension.

    Parameters
    ----------
    locdata
        Localization data
    property_key
        Property key to look for in locdata.

    Returns
    -------
    list[str | None]
        The available property keys in each dimension
    """
    available_keys: list[str | None] = []
    for extension_ in ["_x", "_y", "_z"]:
        prop_key = property_key + extension_
        if prop_key in locdata.columns:
            available_keys.append(prop_key)
        elif property_key in locdata.columns:
            available_keys.append(property_key)
        else:
            available_keys.append(None)
    return available_keys


def _get_linked_coordinates(
    locdata: pd.DataFrame | pd.Series[Any], coordinate_keys: Iterable[str] | None = None
) -> dict[str, int | float | npt.NDArray[np.float64]]:
    """
    Combine localization properties from locdata:
    (i) apply weighted averages for spatial coordinates if corresponding
    uncertainties are available;
    (ii) compute uncertainty (as standard deviation with same units as
    coordinates) from weighted average variances for spatial coordinates.

    If coordinate_keys is None, coordinate_keys and corresponding
    coordinate uncertainties from locdata are used.

    If `uncertainty_x` is not available `uncertainty` is taken.
    If no uncertainty is available, unweighted coordinate means are taken.

    Note
    -----
    Uncertainties should have the same unit as coordinates;
    the weights will be 1 / uncertainties^2.

    Parameters
    ----------
    locdata
        dataframe with locdata
    coordinate_keys
        A selection of coordinate keys on which to compute

    Returns
    -------
    dict[str, int | float | npt.NDArray[np.float64]]
        New position coordinates and related uncertainties as 'uncertainty_c'.
        All values are nan if locdata is empty.
    """
    available_coordinate_keys = _get_loc_property_key_per_dimension(
        locdata=locdata, property_key="position"
    )
    available_uncertainty_keys = _get_loc_property_key_per_dimension(
        locdata=locdata, property_key="uncertainty"
    )
    new_uncertainty_keys = ["uncertainty_x", "uncertainty_y", "uncertainty_z"]

    results_dict = dict()
    for coordinate_key_, uncertainty_key_, new_uncertainty_key_ in zip(
        available_coordinate_keys,
        available_uncertainty_keys,
        new_uncertainty_keys,
    ):
        if coordinate_key_ is not None and (
            coordinate_keys is None or coordinate_key_ in coordinate_keys
        ):
            if not len(locdata):
                weighted_mean = np.nan
                weighted_uncertainty = np.nan
            elif len(locdata) == 1:
                weighted_mean = locdata[coordinate_key_].iloc[0]
                if uncertainty_key_ is None:
                    weighted_uncertainty = 0
                else:
                    weighted_uncertainty = locdata[uncertainty_key_].iloc[0]
            else:
                if uncertainty_key_ is None:
                    weighted_mean, weighted_variance = weighted_mean_variance(
                        values=locdata[coordinate_key_], weights=None
                    )
                    weighted_uncertainty = np.sqrt(weighted_variance)  # type: ignore
                else:
                    if (locdata[uncertainty_key_] == 0).any():
                        logger.warning(
                            "Zero uncertainties occurred resulting in nan for weighted_mean and weighted_variance."
                        )
                    with np.errstate(invalid="ignore"):
                        weighted_mean, weighted_variance = weighted_mean_variance(
                            values=locdata[coordinate_key_],
                            weights=np.power(1 / locdata[uncertainty_key_], 2),
                        )
                        weighted_uncertainty = np.sqrt(weighted_variance)  # type: ignore

            results_dict[coordinate_key_] = weighted_mean
            results_dict[new_uncertainty_key_] = weighted_uncertainty

    return results_dict


def _get_grouped_properties(
    locdata: pd.DataFrame,
    positions: npt.NDArray[np.intp],
    starts: npt.NDArray[np.intp],
    coordinate_keys: Sequence[str],
) -> dict[str, npt.NDArray[Any]] | None:
    """
    Compute the properties of groups of localizations in one pass.

    The properties are those that :class:`locan.LocData` computes for a
    selection of each group: the localization count, coordinates and
    uncertainties as given by :func:`_get_linked_coordinates`,
    the sum of `intensity`, the mean of `local_background`,
    the minimum of `frame` and the measures of the bounding box.

    Parameters
    ----------
    locdata
        dataframe with locdata
    positions
        Row positions in locdata, ordered such that the rows of each group
        are adjacent.
    starts
        Position in `positions` at which each group starts.
    coordinate_keys
        Coordinate keys that span the bounding box.

    Returns
    -------
    dict[str, npt.NDArray[Any]] | None
        Property values with one element per group.
        nan marks a bounding box property that is not available for a
        group.
        None if the properties have to be computed for each group
        separately: there is no group or no coordinate, or a value is not
        numeric or not finite.
    """
    position_keys = _get_loc_property_key_per_dimension(
        locdata=locdata, property_key="position"
    )
    uncertainty_keys = _get_loc_property_key_per_dimension(
        locdata=locdata, property_key="uncertainty"
    )
    used_keys = {
        key
        for key in (
            *position_keys,
            *uncertainty_keys,
            *coordinate_keys,
            "intensity",
            "local_background",
            "frame",
        )
        if key is not None and key in locdata.columns
    }
    dtypes = locdata.dtypes
    if (
        not len(starts)
        or not coordinate_keys
        or not all(
            isinstance(dtypes[key], np.dtype) and dtypes[key].kind in "iuf"
            for key in used_keys
        )
    ):
        return None

    counts = np.diff(np.append(starts, len(positions)))
    is_single = counts == 1

    def take(key: str) -> npt.NDArray[np.float64]:
        values: npt.NDArray[np.float64] = locdata[key].to_numpy(dtype=np.float64)
        return values[positions]

    properties: dict[str, npt.NDArray[Any]] = {"localization_count": counts}

    # divisions by zero are resolved below: values for groups of one
    # localization are set explicitly, other non-finite results return None
    with np.errstate(divide="ignore", invalid="ignore"):
        for position_key, uncertainty_key, new_uncertainty_key in zip(
            position_keys,
            uncertainty_keys,
            ["uncertainty_x", "uncertainty_y", "uncertainty_z"],
        ):
            if position_key is None:
                continue
            values = take(position_key)
            if uncertainty_key is None:
                weights = np.ones(len(values))
                single_uncertainty = np.zeros(len(starts))
            else:
                uncertainties = take(uncertainty_key)
                weights = 1 / uncertainties**2
                single_uncertainty = uncertainties[starts]
            sum_weights = np.add.reduceat(weights, starts)
            mean = np.where(
                is_single,
                values[starts],
                np.add.reduceat(weights * values, starts) / sum_weights,
            )
            # weighted_mean_variance expands this sum of squares into three
            # terms that cancel for coordinates far from zero; the results
            # agree only within that rounding error.
            deviations = weights * (values - np.repeat(mean, counts))
            variance = (
                counts
                / (counts - 1)
                / sum_weights**2
                * np.add.reduceat(deviations**2, starts)
            )
            properties[position_key] = mean
            properties[new_uncertainty_key] = np.where(
                is_single, single_uncertainty, np.sqrt(variance)
            )

        if "intensity" in locdata.columns:
            intensity = locdata["intensity"].to_numpy()[positions]
            # pandas sums integers of any size as 64-bit integers
            dtype = {"i": np.int64, "u": np.uint64}.get(intensity.dtype.kind)
            properties["intensity"] = np.add.reduceat(intensity, starts, dtype=dtype)

        if "local_background" in locdata.columns:
            properties["local_background"] = (
                np.add.reduceat(take("local_background"), starts) / counts
            )

        if "frame" in locdata.columns:
            properties["frame"] = np.minimum.reduceat(
                locdata["frame"].to_numpy()[positions], starts
            )

        widths = []
        for coordinate_key in coordinate_keys:
            values = take(coordinate_key)
            widths.append(
                np.maximum.reduceat(values, starts)
                - np.minimum.reduceat(values, starts)
            )
        width = np.stack(widths, axis=1)
        region_measure = np.where(is_single, 0.0, np.prod(width, axis=1))
        subregion_measure = np.where(is_single, 0.0, 2 * np.sum(width, axis=1))
        properties["region_measure_bb"] = region_measure

        if not all(np.isfinite(values_).all() for values_ in properties.values()):
            return None

        properties["localization_density_bb"] = np.where(
            region_measure != 0, counts / region_measure, np.nan
        )
        properties["subregion_measure_bb"] = np.where(
            subregion_measure != 0, subregion_measure, np.nan
        )

    return properties


def _get_grouped_unique_counts(
    values: npt.NDArray[Any], starts: npt.NDArray[np.intp]
) -> npt.NDArray[np.intp]:
    """
    Count the unique values in each group of adjacent values.

    Parameters
    ----------
    values
        Values, ordered such that the values of each group are adjacent.
    starts
        Position in `values` at which each group starts.

    Returns
    -------
    npt.NDArray[np.intp]
        Number of unique values with one element per group.
    """
    counts = np.diff(np.append(starts, len(values)))
    groups = np.repeat(np.arange(len(starts)), counts)
    sorted_values = values[np.lexsort((values, groups))]
    is_new = np.ones(len(values), dtype=np.intp)
    is_new[1:] = sorted_values[1:] != sorted_values[:-1]
    is_new[starts] = 1
    return_value: npt.NDArray[np.intp] = np.add.reduceat(is_new, starts)
    return return_value


def _bump_property_key(
    loc_property: str, loc_properties: Iterable[str], extension: str = "_0"
) -> str:
    """
    Add extension to loc_property if loc_property is in loc_properties.
    Repeat recursively until loc_property and all loc_properties are unique.
    """
    if loc_property in loc_properties:
        new_property_label = _bump_property_key(
            loc_property=loc_property + extension,
            loc_properties=loc_properties,
            extension=extension,
        )
    else:
        new_property_label = loc_property
    return new_property_label


def _check_loc_properties(
    locdata: LocData, loc_properties: str | Iterable[str] | None
) -> list[str]:
    """
    Check that loc_properties are valid property keys in locdata.

    Parameters
    ----------
    locdata
        Localization data
    loc_properties
        LocData property keys.
        If None the coordinate_keys of locdata are used.

    Returns
    -------
    list[str]
        Valid localization property keys
    """
    if loc_properties is None:  # use coordinate_keys
        labels = locdata.coordinate_keys.copy()
    elif isinstance(loc_properties, str):
        if loc_properties not in locdata.data.columns:
            raise ValueError(
                f"{loc_properties} is not a valid property in locdata.data."
            )
        labels = [loc_properties]
    else:
        labels = []
        for loc_property in loc_properties:
            if loc_property not in locdata.data.columns:
                raise ValueError(
                    f"{loc_property} is not a valid property in locdata.data."
                )
            labels.append(loc_property)
    return labels


def _dataframe_to_pandas(
    dataframe: pd.DataFrame | DataFrame | None, allow_copy: bool = True
) -> pd.DataFrame | None:
    """
    Convert dataframe that supports the dataframe interchange protocol to
    pandas dataframe.
    """
    if isinstance(dataframe, pd.DataFrame) or dataframe is None:
        return_value = dataframe
    elif not hasattr(pd.api, "interchange"):
        msg = (
            "Use pandas version that implements the DataFrame interchange protocol"
            "or provide pandas.DataFrame object."
        )
        raise TypeError(msg)
    else:
        return_value = pd.api.interchange.from_dataframe(
            dataframe, allow_copy=allow_copy
        )
    return return_value

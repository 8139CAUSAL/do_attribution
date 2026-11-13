"""
do_attribution.baselines
------------------------

Single-touch reference allocations: first-touch and last-touch attribution.

These are the naive rules practitioners most often use.  Each converting
journey gives its whole conversion to exactly one channel: the first channel
it touched (:class:`FirstTouchAttribution`) or the last one
(:class:`LastTouchAttribution`).  They are fixed reference points, not
statistical models, so a disagreement between the Markov and Shapley models
can be read against them in :class:`~do_attribution.compare.CompareAttribution`.

Both follow the conventions of the built-in models: the same ``fit`` /
``get_attribution`` / ``summary`` API, the shared validation of journeys and
binary outcomes, conversion-endpoint stripping, and the shared normalization
contract, under which ``attribution`` is ``None`` when no conversion can be
credited to a channel.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .base import BaseAttribution
from .normalization import normalize_signed_attribution
from .reporting import build_path_diagnostics_section
from .utils import (
    DEFAULT_SEPARATOR,
    extract_channels,
    prepare_journey_data,
    split_path,
    strip_conversion_labels_array,
)


class _SingleTouchAttribution(BaseAttribution):
    """Shared implementation of the first-touch and last-touch baselines."""

    #: Position of the credited touch within a journey's channels.
    _touch_index: int = 0
    #: Human-readable name of the rule, used in the rendered summary.
    _touch_name: str = "First-touch"

    def __init__(
        self,
        separator: str = DEFAULT_SEPARATOR,
        conversion_labels: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__(separator=separator, conversion_labels=conversion_labels)

    def fit(
        self,
        paths: Union[Sequence[str], np.ndarray, pd.Series],
        conversions: Union[Sequence[int], np.ndarray, pd.Series],
    ) -> Dict[str, Any]:
        """Credit every converting journey to its single credited touch.

        Returns
        -------
        dict
            Keys:
              - ``'attribution'`` : dict[str, float] of shares summing to 1,
                **or** ``None`` when no conversion can be credited to a
                channel.  Always check before indexing.
              - ``'credited_conversions'`` : dict[str, float] with the number
                of converting journeys credited to each channel.  Always
                present; every fitted channel has an entry, ``0.0`` included.
              - ``'normalization_status'``, ``'normalization_reason'``,
                ``'normalization_denominator'``, ``'raw_sum'``,
                ``'raw_magnitude'``, ``'cancellation_ratio'`` : the shared
                normalization diagnostics, populated in both states.

        Notes
        -----
        A converting journey with no channel (an empty path, or one holding
        only conversion-endpoint or reserved tokens) credits nobody, so
        ``raw_sum`` is the number of conversions on journeys that touched at
        least one channel.
        """
        # Shared preparation path: container shape and binary outcomes are
        # validated on the supplied values, before any coercion.
        paths_arr, conv_arr = prepare_journey_data(paths, conversions)
        if self.conversion_labels:
            paths_arr = strip_conversion_labels_array(
                paths_arr, self.conversion_labels, self.separator
            )
        self.channels = extract_channels(paths_arr, self.separator)

        # A journey's touches are its channels in ``self.channels``, so a
        # reserved token that survived in a path is never credited.
        players = set(self.channels)
        credited: Dict[str, float] = {channel: 0.0 for channel in self.channels}
        for path, converted in zip(paths_arr, conv_arr):
            if not converted:
                continue
            touches = [
                token
                for token in split_path(str(path), self.separator)
                if token in players
            ]
            if touches:
                credited[touches[self._touch_index]] += 1.0

        # The same policy as the Markov and Shapley models: shares exist only
        # when the credited conversions form a valid share vector.
        normalization = normalize_signed_attribution(credited)
        attribution = normalization.attribution
        self._last_attribution = attribution

        result: Dict[str, Any] = {
            "attribution": attribution,
            "credited_conversions": credited,
        }
        result.update(normalization.diagnostics())

        ranked = sorted(credited.items(), key=lambda item: (-item[1], item[0]))
        details = (
            (
                f"{self._touch_name} credited conversions",
                ("Channel", "Conversions"),
                tuple((channel, str(int(count))) for channel, count in ranked),
            ),
        ) if ranked else ()
        details = details + (
            build_path_diagnostics_section(paths_arr, self.separator),
        )
        self._store_static_reporting(
            paths_arr,
            conv_arr,
            model_details=details,
            attribution_note=(
                None
                if normalization.available
                else "Normalized attribution unavailable "
                f"({normalization.normalization_reason}); no conversion was "
                "credited to a channel"
            ),
        )
        return result


class FirstTouchAttribution(_SingleTouchAttribution):
    """First-touch attribution: the first channel gets the whole conversion.

    Parameters
    ----------
    separator : str, default=' > '
        Delimiter used to split path strings into channels.
    conversion_labels : iterable of str, optional
        Tokens in the path strings that represent the conversion endpoint
        (e.g. ``"Purchase"``) rather than a real channel.  Stripped before
        any touch is credited.  See
        :class:`do_attribution.base.BaseAttribution` for full semantics.

    Attributes
    ----------
    channels : list[str]
        All user-visible channels observed across the paths (sorted).
    """

    _touch_index = 0
    _touch_name = "First-touch"


class LastTouchAttribution(_SingleTouchAttribution):
    """Last-touch attribution: the last channel gets the whole conversion.

    Parameters
    ----------
    separator : str, default=' > '
        Delimiter used to split path strings into channels.
    conversion_labels : iterable of str, optional
        Tokens in the path strings that represent the conversion endpoint
        (e.g. ``"Purchase"``) rather than a real channel.  Stripped before
        any touch is credited, so the endpoint is never the last touch.  See
        :class:`do_attribution.base.BaseAttribution` for full semantics.

    Attributes
    ----------
    channels : list[str]
        All user-visible channels observed across the paths (sorted).
    """

    _touch_index = -1
    _touch_name = "Last-touch"


__all__ = ["FirstTouchAttribution", "LastTouchAttribution"]

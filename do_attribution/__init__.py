"""
do_attribution
===================

Multi-touch attribution for Python.  Exposes two statistical allocation
models -- an absorbing-Markov-chain removal-effect model and a
cooperative-game-theoretic Shapley-value model -- through a single
:class:`BaseAttribution` interface, so callers can switch models or
fit both and compare them without changing their call site.

Quick start
-----------

>>> from do_attribution import MarkovAttribution, ShapleyAttribution, CompareAttribution
>>> paths = ['A > B > C', 'A > C', 'B > C']
>>> conversions = [1, 0, 1]
>>> MarkovAttribution(order=1).fit(paths, conversions)['attribution']      # doctest: +SKIP
>>> ShapleyAttribution().fit(paths, conversions)['attribution']             # doctest: +SKIP
>>> CompareAttribution().fit(paths, conversions)                            # doctest: +SKIP
"""

from importlib.metadata import version

from .base import BaseAttribution
from .markov import MarkovAttribution
from .shapley import ShapleyAttribution
from .baselines import FirstTouchAttribution, LastTouchAttribution
from .compare import CompareAttribution
# MODIFIED: export TemporalShapleyAttribution now that the module lives
# inside the package (previously a workspace-root draft).
from .temporal_shapley import TemporalShapleyAttribution
# MODIFIED (temporal v1 punch list): export TemporalMarkovAttribution
# alongside its Shapley sibling, plus the suggest_freq diagnostic.
from .temporal_markov import TemporalMarkovAttribution
from .temporal_diagnostics import suggest_freq
from .temporal import (
    BoundaryEvent,
    DaySummary,
    DecompositionResult,
    EpochRecommendation,
    EpochSufficiencyError,
    EpochWidthError,
    MergeAction,
    NonCalendarBoundaryError,
    PinAction,
    PinOverlapError,
    SegmentFit,
    SignalContinuityError,
    SignalShapeError,
    SignalSufficiencyError,
    StreamingEpochRecommender,
    TemporalRecommendationError,
    WeightedAffineCost,
    assign_paths_to_epochs,
    recommend_epochs,
)
from .reporting import TemporalReportingSnapshot
# MODIFIED: CAL-089 package-root exports for the calibration module.
# Exactly the seven core records/results and both calibrator classes from
# the calibration plan are exported here.  The calibration type aliases
# (Entity, Epoch, RatioPolicy, ...) stay importable from
# ``do_attribution.calibration`` only: no other module exports type
# aliases at the package root, so package conventions do not support
# alias re-export.
from .calibration import (
    AttributionCalibration,
    CalibrationResult,
    CalibrationRow,
    EpochRatioRow,
    EvidencePointRecord,
    LiftTestEvidence,
    RatioScheduleRow,
    TemporalAttributionCalibration,
    TemporalCalibrationResult,
)
from .utils import (
    DEFAULT_SEPARATOR,
    SPECIAL_TOKENS,
    create_n_order_states,
    extract_channels,
    get_powerset,
    prepare_journey_data,
    split_path,
    strip_conversion_labels,  # MODIFIED: P0 export.
    strip_conversion_labels_array,  # MODIFIED: P0 export.
    to_int_array,
    to_string_array,
    validate_data,
)

__version__ = version("do-attribution")

__all__ = [
    # Classes
    "BaseAttribution",
    "MarkovAttribution",
    "ShapleyAttribution",
    "FirstTouchAttribution",
    "LastTouchAttribution",
    "CompareAttribution",
    # MODIFIED: temporal classes exported alongside their static siblings.
    "TemporalShapleyAttribution",
    "TemporalMarkovAttribution",
    # MODIFIED: temporal diagnostics.
    "suggest_freq",
    # Global-signal temporal recommender.
    "BoundaryEvent",
    "DaySummary",
    "DecompositionResult",
    "EpochRecommendation",
    "EpochSufficiencyError",
    "EpochWidthError",
    "MergeAction",
    "NonCalendarBoundaryError",
    "PinAction",
    "PinOverlapError",
    "SegmentFit",
    "SignalContinuityError",
    "SignalShapeError",
    "SignalSufficiencyError",
    "StreamingEpochRecommender",
    "TemporalRecommendationError",
    "TemporalReportingSnapshot",
    "WeightedAffineCost",
    "assign_paths_to_epochs",
    "recommend_epochs",
    # MODIFIED: CAL-089 calibration exports (seven core records/results
    # and both calibrator classes; no type aliases at the package root).
    "LiftTestEvidence",
    "EvidencePointRecord",
    "RatioScheduleRow",
    "EpochRatioRow",
    "CalibrationRow",
    "CalibrationResult",
    "TemporalCalibrationResult",
    "AttributionCalibration",
    "TemporalAttributionCalibration",
    # Utils
    "DEFAULT_SEPARATOR",
    "SPECIAL_TOKENS",
    "create_n_order_states",
    "extract_channels",
    "get_powerset",
    "prepare_journey_data",
    "split_path",
    "strip_conversion_labels",  # MODIFIED: P0 export.
    "strip_conversion_labels_array",  # MODIFIED: P0 export.
    "to_int_array",
    "to_string_array",
    "validate_data",
]

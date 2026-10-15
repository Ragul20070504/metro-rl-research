"""Frozen observation layout for the RL environment (protocol section 6.1).

The layout is defined **once**, here, as an ordered list of :class:`FeatureBlock` descriptors.
Everything else -- the observation vector, its Gymnasium space, the normalisation scaler, the
leakage tests and `docs/rl_environment.md` -- is derived from that list. A hand-maintained table
in the documentation would drift from the code within a week; a generated one cannot.

Each block carries the metadata the research brief requires for every state feature: what it
means, its units, where the value comes from, **when it becomes available**, whether it is an
observed quantity or a forecast, and whether it carries uncertainty.

The `time_availability` field is the one that matters scientifically. Every block must be
computable from simulator state at or before the decision instant `t`; `tests/test_leakage.py`
asserts this empirically by perturbing the future and requiring the observation to be unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.common.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "FeatureBlock",
    "ObservationSpec",
    "ObservationScaler",
    "SIGMA_BLOCK_NAMES",
]

#: Blocks whose *content* differs between the three uncertainty arms (protocol section 6.2).
#: Their shape is identical in every arm, so architecture and parameter count are unchanged.
SIGMA_BLOCK_NAMES = ("forecast_sigma_h1", "forecast_sigma_line_h2h3")

Source = Literal["forecast", "simulator", "clock", "scenario"]


@dataclass(frozen=True)
class FeatureBlock:
    """One contiguous run of observation values sharing a meaning and unit.

    Attributes:
        name: Stable identifier; also the prefix of the per-element feature names.
        size: Number of scalar values.
        meaning: What the values are.
        units: Physical units, or ``dimensionless``.
        source: Which subsystem produces the value.
        time_availability: **The leakage-critical field.** When the value becomes knowable.
        is_forecast: True if the value is a model prediction rather than a realised observation.
        carries_uncertainty: True if the block *is* an uncertainty channel.
        normalised: Whether the affine scaler touches this block.
        element_labels: Optional per-element labels, e.g. station ids.
    """

    name: str
    size: int
    meaning: str
    units: str
    source: Source
    time_availability: str
    is_forecast: bool = False
    carries_uncertainty: bool = False
    normalised: bool = True
    element_labels: tuple[str, ...] = ()

    def feature_names(self) -> list[str]:
        if self.size == 1:
            return [self.name]
        if self.element_labels and len(self.element_labels) == self.size:
            return [f"{self.name}[{label}]" for label in self.element_labels]
        return [f"{self.name}[{i}]" for i in range(self.size)]


@dataclass(frozen=True)
class ObservationSpec:
    """The frozen observation layout for a given network and channel configuration.

    The block order reproduces protocol section 6.1 items 1-13 exactly and must not be
    reordered: `models/obs_scaler.json` and every trained policy are indexed by position.
    """

    blocks: tuple[FeatureBlock, ...]
    n_stations: int
    n_segments: int
    include_availability_channel: bool

    @property
    def size(self) -> int:
        return sum(block.size for block in self.blocks)

    @property
    def feature_names(self) -> list[str]:
        return [name for block in self.blocks for name in block.feature_names()]

    def slice_of(self, name: str) -> slice:
        """Index range of a named block, for tests and for the arm transforms."""
        offset = 0
        for block in self.blocks:
            if block.name == name:
                return slice(offset, offset + block.size)
            offset += block.size
        raise KeyError(f"no observation block named {name!r}")

    def block(self, name: str) -> FeatureBlock:
        for block in self.blocks:
            if block.name == name:
                return block
        raise KeyError(f"no observation block named {name!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "n_stations": self.n_stations,
            "n_segments": self.n_segments,
            "include_availability_channel": self.include_availability_channel,
            "blocks": [
                {
                    "name": b.name,
                    "size": b.size,
                    "meaning": b.meaning,
                    "units": b.units,
                    "source": b.source,
                    "time_availability": b.time_availability,
                    "is_forecast": b.is_forecast,
                    "carries_uncertainty": b.carries_uncertainty,
                    "normalised": b.normalised,
                }
                for b in self.blocks
            ],
        }


def build_observation_spec(
    *,
    station_ids: tuple[str, ...],
    n_segments: int,
    include_availability_channel: bool,
) -> ObservationSpec:
    """Construct the frozen layout of protocol section 6.1.

    Args:
        station_ids: Station labels in line order; length fixes ``N_stations``.
        n_segments: ``K`` frozen line segments for the rolling-stock position summary.
        include_availability_channel: Protocol section 6.1 item 8 is *present only when the
            availability channel is enabled*. Disabling it therefore changes the observation
            dimensionality -- see the H7 confound noted in ``docs/rl_environment.md``.
    """
    n = len(station_ids)
    labels = tuple(station_ids)
    at_t = "known at decision time t (realised at or before t)"
    forecast_at_t = "issued at t from history strictly earlier than t"

    blocks: list[FeatureBlock] = [
        FeatureBlock(
            name="forecast_mu_h1", size=n,
            meaning="per-station forecast mean boardings for the next hour",
            units="passengers/hour", source="forecast",
            time_availability=forecast_at_t, is_forecast=True,
            element_labels=labels,
        ),
        FeatureBlock(
            name="forecast_sigma_h1", size=n,
            meaning=(
                "per-station forecast dispersion sigma-hat for the next hour; "
                "ARM-DEPENDENT content (0 / shuffled / calibrated)"
            ),
            units="passengers/hour", source="forecast",
            time_availability=forecast_at_t, is_forecast=True, carries_uncertainty=True,
            element_labels=labels,
        ),
        FeatureBlock(
            name="forecast_mu_line_h2h3", size=2,
            meaning="line-aggregate forecast mean at horizons h=2 and h=3",
            units="passengers/hour", source="forecast",
            time_availability=forecast_at_t, is_forecast=True,
        ),
        FeatureBlock(
            name="forecast_sigma_line_h2h3", size=2,
            meaning=(
                "line-aggregate forecast dispersion at h=2 and h=3; ARM-DEPENDENT content"
            ),
            units="passengers/hour", source="forecast",
            time_availability=forecast_at_t, is_forecast=True, carries_uncertainty=True,
        ),
        FeatureBlock(
            name="queue_length", size=n,
            meaning="passengers currently waiting on both platforms of the station",
            units="passengers", source="simulator", time_availability=at_t,
            element_labels=labels,
        ),
        FeatureBlock(
            name="boardings_prev_epoch", size=n,
            meaning="passengers who boarded at this station during the previous decision epoch",
            units="passengers", source="simulator",
            time_availability="realised during [t-1, t); complete at t",
            element_labels=labels,
        ),
        FeatureBlock(
            name="fleet_counts", size=3,
            meaning="number of trainsets idle at depot / in service / unavailable",
            units="trainsets", source="simulator", time_availability=at_t,
        ),
    ]

    if include_availability_channel:
        blocks.append(
            FeatureBlock(
                name="availability_signal", size=2,
                meaning=(
                    "mean and minimum noisy availability probability over idle trainsets. "
                    "SYNTHETIC signal (assumption A14, decision D-04): a noisy readout of a "
                    "latent per-train health state drawn at episode start, which *causes* the "
                    "failure hazard. It is not a prediction of the realised failure and "
                    "carries no future information."
                ),
                units="probability", source="simulator",
                time_availability="drawn at episode start; constant within the episode",
                normalised=False,
            )
        )

    blocks += [
        FeatureBlock(
            name="trains_per_segment", size=2 * n_segments,
            meaning=(
                f"count of in-service trainsets in each of K={n_segments} frozen line segments, "
                "direction 0 then direction 1"
            ),
            units="trainsets", source="simulator", time_availability=at_t,
        ),
        FeatureBlock(
            name="occupancy_summary", size=2,
            meaning="mean and maximum occupancy ratio over in-service trainsets",
            units="dimensionless (load/capacity)", source="simulator",
            time_availability=at_t, normalised=False,
        ),
        FeatureBlock(
            name="time_index", size=3,
            meaning="sin and cos of time of day, plus fraction of decision epochs remaining",
            units="dimensionless", source="clock", time_availability=at_t, normalised=False,
        ),
        FeatureBlock(
            name="disruption_flags", size=3,
            meaning=(
                "active_failure (a trainset was lost earlier today and service has not yet "
                "recovered), n_trains_lost_today, active_surge. All three describe events that "
                "have ALREADY occurred or an elevated-demand window the operator is ALREADY in; "
                "none announces a future event."
            ),
            units="indicator / count / indicator", source="scenario",
            time_availability="set by events realised at or before t",
        ),
        FeatureBlock(
            name="headway_state", size=2,
            meaning=(
                "seconds since the last induction, and the smallest headway observed so far "
                "today between consecutive departures anywhere on the line"
            ),
            units="seconds", source="simulator", time_availability=at_t,
        ),
    ]

    spec = ObservationSpec(
        blocks=tuple(blocks),
        n_stations=n,
        n_segments=n_segments,
        include_availability_channel=include_availability_channel,
    )
    logger.info("observation spec: %d blocks, %d values", len(spec.blocks), spec.size)
    return spec


@dataclass
class ObservationScaler:
    """Frozen affine observation scaling (protocol section 6.1).

    ``(x - centre) / scale``. Constants are fitted on **`train`-period days only** and stored in
    ``models/obs_scaler.json``; they are never recomputed on `val` or `test`, and SB3's
    ``VecNormalize`` running statistics are deliberately not used for observations, because a
    running normaliser would adapt differently under each arm and confound the comparison.

    An unfitted scaler is the identity. That is legitimate for smoke and unit tests, but
    :meth:`require_fitted` lets experiment code refuse to run on one.
    """

    centre: np.ndarray
    scale: np.ndarray
    fitted: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def identity(cls, size: int) -> "ObservationScaler":
        return cls(centre=np.zeros(size), scale=np.ones(size), fitted=False,
                   provenance={"status": "identity", "note": "not fitted; smoke/test use only"})

    @classmethod
    def fit(
        cls,
        samples: np.ndarray,
        spec: ObservationSpec,
        provenance: dict[str, Any],
    ) -> "ObservationScaler":
        """Fit centre and scale from observations collected on `train`-period days.

        Blocks flagged ``normalised=False`` -- probabilities, ratios, sin/cos -- are already on a
        sane scale and are left alone, so that a policy's view of them stays interpretable.
        """
        if samples.ndim != 2 or samples.shape[1] != spec.size:
            raise ValueError(
                f"expected samples of shape (n, {spec.size}), got {samples.shape}"
            )
        centre = samples.mean(axis=0)
        scale = samples.std(axis=0)
        # A constant channel has zero spread; dividing by it would produce NaN, and the channel
        # carries no information anyway, so leave it untouched.
        scale = np.where(scale < 1e-8, 1.0, scale)

        for block in spec.blocks:
            if not block.normalised:
                index = spec.slice_of(block.name)
                centre[index] = 0.0
                scale[index] = 1.0

        return cls(centre=centre, scale=scale, fitted=True, provenance=dict(provenance))

    def transform(self, observation: np.ndarray) -> np.ndarray:
        return (observation - self.centre) / self.scale

    def project(self, source: ObservationSpec, target: ObservationSpec) -> "ObservationScaler":
        """Restrict a scaler fitted on ``source`` to the blocks present in ``target``.

        The availability channel is *absent* from the observation when it is disabled, so the
        `ppo_uq_noavail` arm sees 121 values where every other arm sees 123 and a scaler fitted on
        the full layout cannot be applied to it.

        Refitting a separate scaler for that arm would be the wrong repair. The H7 ablation must
        differ from its comparator **only** by the missing channel; a separately fitted scaler
        would also change the centre and scale of all 121 surviving features, so any difference in
        outcome could be attributed to the scaling rather than to the information. Copying each
        retained block's constants across keeps every surviving feature scaled exactly as it is in
        the full-observation arms.

        Raises:
            ValueError: If ``target`` contains a block ``source`` does not, or a shared block
                changed size -- either means the layouts are not nested and slicing would silently
                misalign the constants.
        """
        if not self.fitted:
            return ObservationScaler.identity(target.size)

        source_blocks = {block.name: block for block in source.blocks}
        centre = np.empty(target.size, dtype=float)
        scale = np.empty(target.size, dtype=float)

        for block in target.blocks:
            origin = source_blocks.get(block.name)
            if origin is None:
                raise ValueError(
                    f"block {block.name!r} is absent from the fitted layout, so its scaling "
                    "constants were never measured; the two layouts are not nested"
                )
            if origin.size != block.size:
                raise ValueError(
                    f"block {block.name!r} is {block.size} values in the target layout but "
                    f"{origin.size} in the fitted one; slicing would misalign every later block"
                )
            centre[target.slice_of(block.name)] = self.centre[source.slice_of(block.name)]
            scale[target.slice_of(block.name)] = self.scale[source.slice_of(block.name)]

        dropped = [b.name for b in source.blocks if b.name not in
                   {t.name for t in target.blocks}]
        provenance = dict(self.provenance)
        provenance.update({
            "projected_from_size": source.size,
            "projected_to_size": target.size,
            "dropped_blocks": dropped,
            "note": (
                "constants copied per block from the frozen scaler, so every retained feature is "
                "scaled identically to the full-observation arms; nothing was refitted"
            ),
        })
        return ObservationScaler(centre=centre, scale=scale, fitted=True, provenance=provenance)

    def require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError(
                "observation scaler is unfitted (identity). Reportable runs require frozen "
                "constants fitted on train-period days; run scripts/fit_obs_scaler.py first."
            )

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "centre": self.centre.tolist(),
            "scale": self.scale.tolist(),
            "fitted": self.fitted,
            "provenance": self.provenance,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "ObservationScaler":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            centre=np.asarray(payload["centre"], dtype=float),
            scale=np.asarray(payload["scale"], dtype=float),
            fitted=bool(payload["fitted"]),
            provenance=payload.get("provenance", {}),
        )

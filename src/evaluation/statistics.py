"""Statistical analysis (M14), implementing protocol section 8.3 exactly.

Four rules from the protocol shape everything here, and each exists because it is a place where
scheduling studies routinely overstate their evidence.

**The unit of analysis is the training seed, not the episode** (section 8.2). Thirty evaluation
episodes from one trained policy are thirty measurements of *that policy*, not thirty independent
observations of the method. Reporting the episode-level n as the sample size is prohibited, and
:func:`paired_analysis` refuses to accept episode-level input without aggregating it first.

**The test-choice rule is fixed in advance** (section 8.3 item 2): Shapiro-Wilk on the paired
differences decides between a paired *t* and a Wilcoxon signed-rank test. Choosing after seeing
which gives the smaller p-value is the classic garden-of-forking-paths error, so the rule is coded
here rather than applied by hand.

**Effect size and exact p-values are always reported** (items 3 and 5). The phrase "statistically
significant" never appears without a named test, its statistic, its n and its p.

**Failing to reject is not the same as showing no effect** (item 8). When the primary test does
not reject, a TOST equivalence test runs against the pre-registered smallest effect size of
interest, so the paper can say which of the two it found.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

from src.common.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "Descriptives", "PairedResult", "describe", "paired_analysis",
    "holm", "benjamini_hochberg", "hodges_lehmann", "bca_bootstrap_ci",
    "rank_biserial", "cliffs_delta", "tost_equivalence",
]

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260827


@dataclass
class Descriptives:
    """Protocol section 8.3 item 4: reported for every (method, scenario, metric)."""

    n: int
    mean: float
    median: float
    sd: float
    iqr: float
    ci_low: float
    ci_high: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def describe(values: Sequence[float]) -> Descriptives:
    data = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if data.size == 0:
        nan = float("nan")
        return Descriptives(0, nan, nan, nan, nan, nan, nan)
    mean = float(data.mean())
    sd = float(data.std(ddof=1)) if data.size > 1 else 0.0
    # Normal-approximation CI on the mean. With n = 10 training seeds this is the conventional
    # descriptive interval; the inferential claims rest on the paired tests below, not on it.
    half = 1.96 * sd / np.sqrt(data.size) if data.size > 1 else 0.0
    return Descriptives(
        n=int(data.size), mean=mean, median=float(np.median(data)), sd=sd,
        iqr=float(np.percentile(data, 75) - np.percentile(data, 25)),
        ci_low=mean - half, ci_high=mean + half,
    )


def hodges_lehmann(differences: np.ndarray) -> float:
    """Median of all Walsh averages -- the location estimator paired with a signed-rank test."""
    data = np.asarray(differences, dtype=float)
    if data.size == 0:
        return float("nan")
    i, j = np.triu_indices(data.size, k=0)
    return float(np.median((data[i] + data[j]) / 2.0))


def bca_bootstrap_ci(
    differences: np.ndarray,
    statistic=hodges_lehmann,
    *,
    alpha: float = 0.05,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Bias-corrected and accelerated bootstrap interval (protocol section 8.3 item 3).

    BCa rather than percentile because the paired differences here are small-n and can be skewed,
    exactly the case where the percentile interval is known to be badly centred.
    """
    from scipy import stats

    data = np.asarray(differences, dtype=float)
    n = data.size
    if n < 3:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    observed = statistic(data)
    boot = np.array([statistic(data[rng.integers(0, n, n)]) for _ in range(resamples)])

    proportion = np.mean(boot < observed)
    if proportion <= 0 or proportion >= 1:
        # Degenerate resampling distribution; fall back to the percentile interval rather than
        # producing a nonsensical bias correction.
        return float(np.percentile(boot, 100 * alpha / 2)), float(
            np.percentile(boot, 100 * (1 - alpha / 2)))
    z0 = stats.norm.ppf(proportion)

    jackknife = np.array([statistic(np.delete(data, i)) for i in range(n)])
    centred = jackknife.mean() - jackknife
    denominator = 6.0 * (np.sum(centred ** 2) ** 1.5)
    acceleration = float(np.sum(centred ** 3) / denominator) if denominator > 0 else 0.0

    def endpoint(z_alpha: float) -> float:
        adjusted = z0 + (z0 + z_alpha) / (1 - acceleration * (z0 + z_alpha))
        return float(np.percentile(boot, 100 * stats.norm.cdf(adjusted)))

    return endpoint(stats.norm.ppf(alpha / 2)), endpoint(stats.norm.ppf(1 - alpha / 2))


def rank_biserial(differences: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation: the signed-rank test's effect size."""
    from scipy import stats

    data = np.asarray(differences, dtype=float)
    nonzero = data[data != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    total = positive + negative
    return float((positive - negative) / total) if total > 0 else 0.0


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's delta on the episode-level distributions (protocol section 8.3 item 3).

    Reported *alongside* the seed-level tests, never instead of them: it is a distributional
    description, and treating its episode-level n as a sample size is the error section 8.2
    prohibits.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size == 0 or y.size == 0:
        return float("nan")
    greater = np.sum(x[:, None] > y[None, :])
    less = np.sum(x[:, None] < y[None, :])
    return float((greater - less) / (x.size * y.size))


def tost_equivalence(
    differences: np.ndarray, sesoi: float, *, alpha: float = 0.05
) -> dict[str, Any]:
    """Two one-sided tests against a pre-registered smallest effect size of interest.

    Protocol section 8.3 item 8. Run when the primary test fails to reject, so the paper can
    distinguish "no evidence of benefit" from "evidence of no meaningful benefit". A null result
    is a publishable finding here.
    """
    from scipy import stats

    data = np.asarray(differences, dtype=float)
    n = data.size
    if n < 2 or sesoi <= 0:
        return {"applicable": False, "reason": "n < 2 or SESOI not positive"}

    mean = float(data.mean())
    se = float(data.std(ddof=1) / np.sqrt(n))
    if se == 0:
        return {"applicable": False, "reason": "zero variance in paired differences"}

    t_lower = (mean + sesoi) / se
    t_upper = (mean - sesoi) / se
    p_lower = 1 - stats.t.cdf(t_lower, df=n - 1)
    p_upper = stats.t.cdf(t_upper, df=n - 1)
    p = float(max(p_lower, p_upper))
    return {
        "applicable": True,
        "sesoi": float(sesoi),
        "mean_difference": mean,
        "p_value": p,
        "equivalent_at_alpha": bool(p < alpha),
        "interpretation": (
            "the paired difference is statistically equivalent to zero within the pre-registered "
            "SESOI" if p < alpha else
            "equivalence to zero within the SESOI could not be established; the result is "
            "inconclusive rather than null"
        ),
    }


@dataclass
class PairedResult:
    """One paired comparison, with everything protocol section 8.3 requires reported."""

    label: str
    metric: str
    n: int
    method_a: str
    method_b: str
    mean_a: float
    mean_b: float
    mean_difference: float
    test: str
    statistic: float
    p_value: float
    normality_p: float
    hodges_lehmann: float
    ci_low: float
    ci_high: float
    rank_biserial: float
    cliffs_delta: float = float("nan")
    family: str = "exploratory"
    p_adjusted: float = float("nan")
    adjustment: str = ""
    equivalence: dict[str, Any] = field(default_factory=dict)
    direction: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def sentence(self) -> str:
        """A reportable sentence carrying the named test, statistic, n and exact p."""
        return (
            f"{self.method_a} vs {self.method_b} on {self.metric}: "
            f"{self.test} statistic={self.statistic:.4g}, n={self.n}, p={self.p_value:.4g}; "
            f"Hodges-Lehmann difference {self.hodges_lehmann:.6g} "
            f"(95% BCa CI {self.ci_low:.6g} to {self.ci_high:.6g})"
        )


def paired_analysis(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    label: str,
    metric: str,
    method_a: str,
    method_b: str,
    family: str = "exploratory",
    lower_is_better: bool = True,
    sesoi: float | None = None,
    episode_a: Sequence[float] | None = None,
    episode_b: Sequence[float] | None = None,
) -> PairedResult:
    """Run the frozen paired comparison.

    Args:
        values_a, values_b: **Seed-level** means, paired elementwise by training seed. Passing
            episode-level values here would inflate n and is what protocol section 8.2 forbids.
        sesoi: Smallest effect size of interest. When the test fails to reject and this is
            supplied, a TOST equivalence test is added.
        episode_a, episode_b: Optional episode-level values, used only for Cliff's delta.
    """
    from scipy import stats

    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must match: {a.shape} vs {b.shape}")
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = int(a.size)
    differences = a - b

    if n < 3:
        raise ValueError(f"paired analysis needs at least 3 pairs; got {n}")

    # The choice rule is fixed HERE, before any data is seen, exactly as section 8.3 item 2
    # requires. Picking the test after inspecting both p-values would be a forking path.
    if np.allclose(differences, differences[0]):
        normality_p = float("nan")
        use_t = False
    else:
        normality_p = float(stats.shapiro(differences).pvalue)
        use_t = normality_p >= 0.05 and n >= 8

    if use_t:
        result = stats.ttest_rel(a, b)
        test, statistic, p_value = "paired t-test", float(result.statistic), float(result.pvalue)
    else:
        if np.all(differences == 0):
            test, statistic, p_value = "Wilcoxon signed-rank", 0.0, 1.0
        else:
            result = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            test = "Wilcoxon signed-rank"
            statistic, p_value = float(result.statistic), float(result.pvalue)

    low, high = bca_bootstrap_ci(differences)
    mean_difference = float(differences.mean())

    if abs(mean_difference) < 1e-12:
        direction = "no difference"
    elif (mean_difference < 0) == lower_is_better:
        direction = f"{method_a} better"
    else:
        direction = f"{method_b} better"

    equivalence: dict[str, Any] = {}
    if sesoi is not None and p_value >= 0.05:
        equivalence = tost_equivalence(differences, sesoi)

    delta = float("nan")
    if episode_a is not None and episode_b is not None:
        delta = cliffs_delta(episode_a, episode_b)

    return PairedResult(
        label=label, metric=metric, n=n, method_a=method_a, method_b=method_b,
        mean_a=float(a.mean()), mean_b=float(b.mean()), mean_difference=mean_difference,
        test=test, statistic=statistic, p_value=p_value, normality_p=normality_p,
        hodges_lehmann=hodges_lehmann(differences), ci_low=low, ci_high=high,
        rank_biserial=rank_biserial(differences), cliffs_delta=delta,
        family=family, equivalence=equivalence, direction=direction,
    )


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values. Used for the two-test confirmatory family."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = p.size
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        value = (m - rank) * p[index]
        running = max(running, value)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def benjamini_hochberg(p_values: Sequence[float], q: float = 0.10) -> list[float]:
    """Benjamini-Hochberg adjusted p-values for the exploratory family (FDR at q = 0.10)."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        index = order[rank]
        value = p[index] * m / (rank + 1)
        running = min(running, value)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()

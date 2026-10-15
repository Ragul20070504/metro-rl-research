"""Functional smoke test of the research stack (milestone M2).

Importing a package proves it installed; it does not prove it works. This script exercises
the specific capability each dependency was chosen for in ``docs/dependency_plan.md``, so a
broken stack is discovered here rather than at M8 after a week of implementation.

It deliberately does **not** touch project data, models or results. It is safe to run at any
time and produces no artefacts.

Run::

    .venv/Scripts/python.exe scripts/check_environment.py
"""

from __future__ import annotations

try:  # `python scripts/x.py` puts scripts/ on sys.path, not the repository root
    import _path  # noqa: F401
except ModuleNotFoundError:  # imported as `scripts.x`; the root is already on sys.path
    pass

import platform
import sys
import traceback
from collections.abc import Callable

# Each check returns a short string describing what it proved.
Check = tuple[str, Callable[[], str]]


def check_numpy_seeding() -> str:
    """Independent, reproducible RNG streams -- the basis of common random numbers."""
    import numpy as np

    parent = np.random.SeedSequence(12345)
    a1, b1 = (np.random.default_rng(s) for s in parent.spawn(2))
    a2, b2 = (np.random.default_rng(s) for s in np.random.SeedSequence(12345).spawn(2))
    draw_a1, draw_b1 = a1.normal(size=5), b1.normal(size=5)
    assert np.array_equal(draw_a1, a2.normal(size=5)), "spawned stream not reproducible"
    assert np.array_equal(draw_b1, b2.normal(size=5)), "spawned stream not reproducible"
    assert not np.array_equal(draw_a1, draw_b1), "spawned streams are not independent"
    return f"numpy {np.__version__}: SeedSequence spawn reproducible and independent"


def check_pandas_parquet() -> str:
    """Round-trip a frame through parquet -- forecast.parquet and episode logs depend on it."""
    import io

    import pandas as pd

    frame = pd.DataFrame(
        {
            "issue_time": pd.to_datetime(["2024-01-01 08:00", "2024-01-01 09:00"], utc=True),
            "station_id": ["A", "B"],
            "p50": [10.0, 20.0],
            "sigma": [1.5, 2.5],
        }
    )
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    buffer.seek(0)
    restored = pd.read_parquet(buffer)
    assert restored.equals(frame), "parquet round-trip changed the frame"
    return f"pandas {pd.__version__}: parquet round-trip exact"


def check_scipy_paired_tests() -> str:
    """Paired tests and bootstrap -- protocol section 8.3 depends on both."""
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    differences = rng.normal(loc=0.4, scale=1.0, size=20)
    signed_rank = stats.wilcoxon(differences)
    shapiro = stats.shapiro(differences)
    boot = stats.bootstrap(
        (differences,), np.median, confidence_level=0.95, method="BCa",
        n_resamples=999, random_state=0,
    )
    assert 0.0 <= signed_rank.pvalue <= 1.0
    assert 0.0 <= shapiro.pvalue <= 1.0
    assert boot.confidence_interval.low < boot.confidence_interval.high
    return (
        f"scipy {stats.__name__.split('.')[0]}: wilcoxon signed-rank, shapiro, "
        "and BCa bootstrap all functional"
    )


def check_statsmodels_multiplicity() -> str:
    """Holm and Benjamini-Hochberg -- the confirmatory / exploratory split needs both."""
    from statsmodels.stats.multitest import multipletests

    raw = [0.001, 0.02, 0.04, 0.3]
    holm_reject, holm_p, _, _ = multipletests(raw, alpha=0.05, method="holm")
    bh_reject, bh_p, _, _ = multipletests(raw, alpha=0.10, method="fdr_bh")
    assert holm_reject[0] and not holm_reject[-1]
    assert all(adj >= orig for adj, orig in zip(holm_p, raw)), "Holm must not lower a p-value"
    assert bh_reject[0]
    return "statsmodels: Holm and Benjamini-Hochberg corrections functional"


def check_xgboost_quantile() -> str:
    """Pinball-loss quantile regression -- the uncertainty channel comes from this."""
    import numpy as np
    import xgboost as xgb

    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 3))
    y = x[:, 0] * 2.0 + rng.normal(scale=1.0, size=400)
    quantiles = np.array([0.1, 0.5, 0.9])
    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=quantiles,
        n_estimators=40,
        max_depth=3,
        random_state=0,
    )
    model.fit(x, y)
    predictions = model.predict(x)
    assert predictions.shape == (400, 3), f"unexpected quantile output shape {predictions.shape}"
    crossing = (np.diff(predictions, axis=1) < 0).mean()
    assert crossing < 0.10, f"excessive quantile crossing: {crossing:.1%}"
    return (
        f"xgboost {xgb.__version__}: multi-quantile pinball regression fits, "
        f"quantile crossing {crossing:.1%}"
    )


def check_torch_determinism() -> str:
    """Deterministic CPU forward pass -- evaluation reproducibility depends on it."""
    import torch

    def forward() -> torch.Tensor:
        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 3))
        return net(torch.ones(1, 8))

    first, second = forward(), forward()
    assert torch.equal(first, second), "identical seeds produced different outputs"
    return f"torch {torch.__version__}: deterministic seeded forward pass on CPU"


def check_maskable_ppo() -> str:
    """MaskablePPO end to end -- action masking is mandated by the manuscript (section 3.4).

    Uses a throwaway toy environment in which one action is always masked out, and asserts
    the trained policy never selects it. This is the single most important stack check: if
    masking silently fails, invalid dispatches would corrupt every experiment.
    """
    import numpy as np
    from gymnasium import Env, spaces
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    forbidden = 2

    class ToyMaskedEnv(Env):
        """Three actions; action `forbidden` is always infeasible."""

        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        action_space = spaces.Discrete(3)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._step = 0
            return np.zeros(4, dtype=np.float32), {}

        def step(self, action):
            assert action != forbidden, "a masked action was executed"
            self._step += 1
            return (
                np.zeros(4, dtype=np.float32),
                float(action),
                self._step >= 8,
                False,
                {},
            )

        def action_masks(self) -> np.ndarray:
            mask = np.ones(3, dtype=bool)
            mask[forbidden] = False
            return mask

    env = ToyMaskedEnv()
    model = MaskablePPO("MlpPolicy", env, n_steps=64, batch_size=32, verbose=0, seed=0)
    model.learn(total_timesteps=128)

    observation, _ = env.reset()
    chosen = set()
    for _ in range(20):
        action, _ = model.predict(
            observation, action_masks=get_action_masks(env), deterministic=False
        )
        chosen.add(int(action))
    assert forbidden not in chosen, "policy selected a masked action"
    return f"sb3-contrib MaskablePPO: trained 128 steps, masked action never selected"


def check_ortools_mip() -> str:
    """CP-SAT solve with status and bound reporting -- required for the MILP baselines."""
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    trains = [model.NewIntVar(0, 3, f"induct_{t}") for t in range(4)]
    model.Add(sum(trains) <= 6)
    model.Maximize(sum((i + 1) * trains[i] for i in range(4)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0
    status = solver.Solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), "solver did not find a solution"
    assert solver.Value(trains[3]) == 3, "unexpected optimal solution"
    return (
        f"ortools CP-SAT: status={solver.StatusName(status)} "
        f"objective={solver.ObjectiveValue():.0f} wall={solver.WallTime():.3f}s "
        "(status and time are recoverable, as the protocol requires)"
    )


def check_shap() -> str:
    """SHAP attribution over a small model -- the explainability layer depends on it."""
    import numpy as np
    import shap
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 4))
    y = x[:, 0] * 3.0 + x[:, 1]
    model = RandomForestRegressor(n_estimators=15, random_state=0).fit(x, y)
    values = shap.TreeExplainer(model).shap_values(x[:20])
    assert values.shape == (20, 4), f"unexpected shap output shape {values.shape}"
    importance = np.abs(values).mean(axis=0)
    assert importance.argmax() == 0, "shap did not recover the dominant feature"
    return f"shap {shap.__version__}: attributions recovered the dominant feature"


def check_matplotlib_headless() -> str:
    """Figure rendering without a display -- figures are generated in batch, never by hand."""
    import matplotlib

    matplotlib.use("Agg")
    import io

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(2, 2))
    axes.plot([0, 1, 2], [0, 1, 4])
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=50)
    plt.close(figure)
    assert buffer.getbuffer().nbytes > 0, "no image bytes written"
    return f"matplotlib {matplotlib.__version__}: Agg backend renders headless"


CHECKS: list[Check] = [
    ("numpy seeding", check_numpy_seeding),
    ("pandas/parquet", check_pandas_parquet),
    ("scipy paired tests", check_scipy_paired_tests),
    ("statsmodels multiplicity", check_statsmodels_multiplicity),
    ("xgboost quantile", check_xgboost_quantile),
    ("torch determinism", check_torch_determinism),
    ("MaskablePPO", check_maskable_ppo),
    ("ortools CP-SAT", check_ortools_mip),
    ("shap", check_shap),
    ("matplotlib headless", check_matplotlib_headless),
]


def main() -> int:
    """Run every check and return a process exit code (0 = all passed)."""
    print(f"python   {sys.version.split()[0]}  ({sys.executable})")
    print(f"platform {platform.platform()}  {platform.machine()}")
    print()

    failures: list[str] = []
    for name, check in CHECKS:
        try:
            detail = check()
        except Exception:  # noqa: BLE001 - a smoke test must report, not propagate
            failures.append(name)
            print(f"FAIL  {name}")
            print(traceback.format_exc())
        else:
            print(f"PASS  {name:26s} {detail}")

    print()
    if failures:
        print(f"{len(failures)} of {len(CHECKS)} checks FAILED: {', '.join(failures)}")
        return 1
    print(f"all {len(CHECKS)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

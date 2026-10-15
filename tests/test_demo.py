"""The M17 demonstration layer must show the research, not re-implement it.

Three properties make the difference between a demonstration and a plausible-looking mock-up, and
all three fail silently if broken:

- **It uses the real components.** If the dashboard computed its own actions or metrics, it would
  keep working while showing something the experiment never produced.
- **It cannot see the future.** A demonstration that fed realised demand to the policy would look
  impressive and mean nothing.
- **It cannot touch the confirmatory results.** An interactive run writing into `results/runs/`
  would corrupt the evidence for the paper.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.demo import SCENARIOS, DemoSession, available_policies
from src.demo.session import DEMO_RESULTS_DIR, MissingArtifactError


def _first_available() -> str:
    for option in available_policies():
        if option.available and not option.needs_train_seed:
            return option.method_id
    pytest.skip("no baseline artefacts available yet")


@pytest.fixture(scope="module")
def method() -> str:
    return _first_available()


# --------------------------------------------------------------------------- availability


class TestPolicyAvailability:
    def test_every_registered_policy_is_offered_with_a_verdict(self):
        options = available_policies()
        ids = {o.method_id for o in options}
        assert {"fixed_timetable", "heuristic_threshold", "milp_static",
                "milp_rolling", "ppo_point", "ppo_uq"} <= ids

    def test_an_unavailable_policy_states_why(self):
        for option in available_policies():
            if not option.available:
                assert option.reason, f"{option.method_id} is unavailable with no reason given"

    def test_selecting_a_policy_without_an_artefact_fails_loudly(self):
        """The dangerous alternative is a silent fallback to a different policy."""
        unavailable = [o for o in available_policies() if not o.available]
        if not unavailable:
            pytest.skip("every policy has an artefact")
        with pytest.raises(MissingArtifactError):
            DemoSession(unavailable[0].method_id, "normal", eval_seed=1)

    def test_an_unknown_scenario_is_rejected(self, method):
        with pytest.raises(ValueError):
            DemoSession(method, "heatwave", eval_seed=1)

    def test_a_ppo_seed_without_a_trained_policy_is_rejected(self):
        ppo = [o for o in available_policies()
               if o.needs_train_seed and o.available]
        if not ppo:
            pytest.skip("no trained PPO policies yet")
        with pytest.raises(MissingArtifactError):
            DemoSession(ppo[0].method_id, "normal", eval_seed=1, train_seed=999_999)


# --------------------------------------------------------------------------- stepping


class TestStepping:
    def test_stepping_advances_the_actual_simulator(self, method):
        session = DemoSession(method, "normal", eval_seed=3)
        before = session.env.simulator.epoch
        session.step()
        assert session.env.simulator.epoch == before + 1
        assert len(session.history) == 1

    def test_the_action_comes_from_the_policy_and_respects_the_mask(self, method):
        session = DemoSession(method, "surge", eval_seed=3)
        for record in session.run(5):
            assert record.executed_action <= max(record.feasible_actions)
            assert record.action_mask[record.executed_action]
            assert record.executed_action <= record.requested_action

    def test_reset_returns_to_the_start(self, method):
        session = DemoSession(method, "normal", eval_seed=3)
        session.run(4)
        session.reset()
        assert session.history == []
        assert session.env.simulator.epoch == 0

    def test_run_stops_at_the_end_of_the_episode(self, method):
        session = DemoSession(method, "normal", eval_seed=3)
        horizon = session.env.sim_config.n_decision_epochs
        produced = session.run(horizon + 25)
        assert len(produced) <= horizon
        assert session.terminated

    def test_stepping_a_finished_episode_raises(self, method):
        session = DemoSession(method, "normal", eval_seed=3)
        session.run(session.env.sim_config.n_decision_epochs + 5)
        with pytest.raises(RuntimeError):
            session.step()


# --------------------------------------------------------------------------- reproducibility


class TestSeedReproducibility:
    def test_the_same_seed_reproduces_the_same_episode(self, method):
        a = DemoSession(method, "surge", eval_seed=5)
        b = DemoSession(method, "surge", eval_seed=5)
        first = [(r.executed_action, round(r.reward, 9)) for r in a.run(8)]
        second = [(r.executed_action, round(r.reward, 9)) for r in b.run(8)]
        assert first == second

    def test_a_different_seed_gives_a_different_episode(self, method):
        a = DemoSession(method, "surge", eval_seed=5)
        b = DemoSession(method, "surge", eval_seed=6)
        assert [r.reward for r in a.run(8)] != [r.reward for r in b.run(8)]

    def test_reset_reproduces_the_first_run(self, method):
        session = DemoSession(method, "normal", eval_seed=9)
        first = [round(r.reward, 9) for r in session.run(6)]
        session.reset()
        again = [round(r.reward, 9) for r in session.run(6)]
        assert first == again


# --------------------------------------------------------------------------- state readout


class TestStateReadout:
    def test_the_network_is_the_real_one(self, method):
        session = DemoSession(method, "normal", eval_seed=2)
        state = session.network_state()
        assert len(state["stations"]) == session.env.network.n_stations
        assert len(state["trains"]) == state["fleet_size"]
        assert all(s["station_id"] and s["name"] for s in state["stations"])

    def test_train_state_tracks_the_simulator(self, method):
        session = DemoSession(method, "normal", eval_seed=2)
        session.run(8)
        state = session.network_state()
        live = sum(1 for t in state["trains"] if t["in_service"])
        assert live == sum(1 for t in session.env.simulator.trains if t.in_service)
        for train in state["trains"]:
            assert 0.0 <= train["occupancy_fraction"] <= 1.0 + 1e-9

    def test_passenger_queues_track_the_simulator(self, method):
        session = DemoSession(method, "surge", eval_seed=2)
        session.run(10)
        state = session.network_state()
        for station in state["stations"]:
            expected = sum(session.env.simulator.queues[station["index"]][d].size
                           for d in (0, 1))
            assert station["queue"] == pytest.approx(float(expected))

    def test_the_forecast_panel_reports_the_arm_honestly(self, method):
        session = DemoSession(method, "normal", eval_seed=2)
        forecast = session.forecast_state()
        sigma = np.asarray(forecast["sigma_per_station"], dtype=float)
        if forecast["arm"] == "point":
            assert not forecast["carries_uncertainty"]
            assert float(np.abs(sigma).sum()) == 0.0
        assert len(forecast["mu_per_station"]) == session.env.network.n_stations

    def test_the_primary_endpoint_is_displayed(self, method):
        session = DemoSession(method, "surge", eval_seed=2)
        session.run(12)
        primary = session.metrics()["primary_endpoint"]
        assert primary["name"] == "left_behind_fraction"
        assert np.isfinite(primary["value"])

    def test_reward_components_come_from_the_reward_implementation(self, method):
        """The dashboard must not recompute the reward with its own formula."""
        session = DemoSession(method, "surge", eval_seed=2)
        record = session.run(3)[-1]
        assert set(record.reward_components) == {"W_minutes", "U_passengers", "E_units", "C_units"}

        env_record = session.env.episode_records()[-1]
        for key, value in record.reward_components.items():
            assert value == pytest.approx(float(env_record[key]))
        assert record.reward == pytest.approx(float(env_record["reward"]))


# --------------------------------------------------------------------------- no future info


class TestNoFutureInformation:
    def test_realised_demand_is_only_available_after_it_occurs(self, method):
        session = DemoSession(method, "surge", eval_seed=4)
        assert session.realised_demand() == [], "nothing is realised before the first step"
        session.run(5)
        assert len(session.realised_demand()) == 5

    def test_the_forecast_accessor_refuses_the_present_hour(self):
        from src.environment.forecast_provider import HistoryBuffer

        buffer = HistoryBuffer(np.arange(48 * 3, dtype=float).reshape(48, 3))
        with pytest.raises(IndexError):
            buffer.value_at(12, issue_hour=12)

    def test_the_policy_sees_only_the_environment_observation(self, method):
        """The session must hand the policy the env's observation, not something it assembled."""
        session = DemoSession(method, "normal", eval_seed=4)
        seen: list[np.ndarray] = []
        original = session.policy.act

        def recording(observation, env, mask):
            seen.append(np.array(observation, copy=True))
            return original(observation, env, mask)

        session.policy.act = recording  # type: ignore[method-assign]
        session.step()
        assert len(seen) == 1
        assert seen[0].shape == session.env.observation_space.shape


# --------------------------------------------------------------------------- separation


class TestResultSeparation:
    def test_demo_output_goes_to_the_demo_directory(self, method, tmp_path, monkeypatch):
        from src.demo import session as session_module

        target = tmp_path / "demo"
        monkeypatch.setattr(session_module, "DEMO_RESULTS_DIR", target)

        session = DemoSession(method, "normal", eval_seed=8)
        session.run(3)
        path = session.save()

        assert path.parent == target
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["traceability"]["kind"].startswith("INTERACTIVE DEMONSTRATION")
        assert len(payload["steps"]) == 3

    def test_the_demo_directory_is_not_the_confirmatory_tree(self):
        from src.evaluation.runner import RUNS_DIR

        assert DEMO_RESULTS_DIR != RUNS_DIR
        assert RUNS_DIR not in DEMO_RESULTS_DIR.parents
        assert DEMO_RESULTS_DIR not in RUNS_DIR.parents

    def test_demo_cells_carry_a_variant_that_excludes_them_from_analysis(self, method):
        """`paired_frame` reads variant == 'main'; a demo cell must never match it."""
        session = DemoSession(method, "normal", eval_seed=8)
        assert session._cell.variant == "demo"
        assert session._cell.variant != "main"

    def test_a_demo_run_writes_nothing_under_results_runs(self, method, tmp_path, monkeypatch):
        from src.demo import session as session_module
        from src.evaluation.runner import RUNS_DIR

        monkeypatch.setattr(session_module, "DEMO_RESULTS_DIR", tmp_path / "demo")
        before = sorted(p.name for p in RUNS_DIR.rglob("metrics.json")) if RUNS_DIR.exists() else []

        session = DemoSession(method, "surge", eval_seed=8)
        session.run(4)
        session.save()

        after = sorted(p.name for p in RUNS_DIR.rglob("metrics.json")) if RUNS_DIR.exists() else []
        assert before == after


# --------------------------------------------------------------------------- traceability


class TestApplication:
    """The Streamlit layer must import and must not contain a second implementation."""

    def test_the_app_module_imports(self):
        pytest.importorskip("streamlit", reason="the demonstration UI is an optional extra")
        import importlib

        module = importlib.import_module("app")
        for name in ("main", "sidebar", "network_panel", "forecast_panel",
                     "decision_panel", "metrics_panel", "traceability_panel"):
            assert hasattr(module, name), f"app is missing {name}"

    def test_the_app_carries_the_demonstration_disclaimer(self):
        pytest.importorskip("streamlit", reason="the demonstration UI is an optional extra")
        import importlib

        module = importlib.import_module("app")
        banner = module.DEMO_BANNER.lower()
        assert "not confirmatory" in banner
        assert "single draw" in banner or "one episode" in banner

    def test_the_app_does_not_reimplement_decisions_or_rewards(self):
        """A UI that computed its own action or reward could show something never produced."""
        from src.common.paths import REPO_ROOT

        source = (REPO_ROOT / "app.py").read_text(encoding="utf-8")
        for forbidden in ("compute_reward", "action_masks(", "simulator.step_epoch",
                          "def act(", "train_policy"):
            assert forbidden not in source, (
                f"app.py references {forbidden!r}: the UI must render state, not compute it")

    def test_the_demo_package_never_writes_to_the_confirmatory_tree(self):
        from src.common.paths import REPO_ROOT

        source = (REPO_ROOT / "src" / "demo" / "session.py").read_text(encoding="utf-8")
        assert "RUNS_DIR" not in source
        assert "run_cell" not in source, "a demo session must not invoke the confirmatory runner"


class TestTraceability:
    def test_a_run_records_what_is_needed_to_reproduce_it(self, method):
        session = DemoSession(method, "surge", eval_seed=13)
        trace = session.traceability()
        for key in ("method", "scenario", "eval_seed", "git_commit", "n_stations",
                    "horizon_epochs", "fleet_size", "packages", "reward_scales"):
            assert key in trace, f"traceability is missing {key}"
        assert trace["eval_seed"] == 13
        assert "NOT A CONFIRMATORY RESULT" in trace["kind"]

    def test_every_scenario_can_be_constructed(self, method):
        for scenario in SCENARIOS:
            session = DemoSession(method, scenario, eval_seed=1)
            assert session.scenario == scenario
            session.step()


class TestPilotPoliciesAreNotOffered:
    """The R-1 pilot's policies are discarded by protocol section 8.4.

    They exist on disk because the pilot ran, so enumerating the checkpoint directory finds them.
    Offering one in the dashboard would let a viewer watch a discarded policy and read it as the
    study's result, which is a presentation-integrity problem even though demo output is written
    under `results/demo/` and can never reach a confirmatory table.
    """

    def test_no_offered_seed_is_a_pilot_seed(self):
        from src.demo.session import PILOT_SEED_FLOOR, available_policies

        offered = {seed for option in available_policies() for seed in option.train_seeds}
        assert offered, "the demo must offer at least one trained policy"
        assert not [s for s in offered if s >= PILOT_SEED_FLOOR], (
            f"pilot seeds are offered: {sorted(s for s in offered if s >= PILOT_SEED_FLOOR)}")

    def test_the_registered_seeds_are_still_offered(self):
        """The filter must remove the pilot seeds only, not the confirmatory pool with them."""
        from src.demo.session import available_policies

        by_method = {o.method_id: o for o in available_policies()}
        for method in ("ppo_point", "ppo_uq", "ppo_placebo"):
            assert set(by_method[method].train_seeds) == set(range(101, 111)), (
                f"{method} should offer exactly the registered pool 101-110")

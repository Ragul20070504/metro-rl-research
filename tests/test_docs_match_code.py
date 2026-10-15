"""Documentation that describes a frozen interface must not drift from it.

`docs/rl_environment.md` documents the observation layout, the arms, the action space and the
reward. A hand-maintained table drifts from the code within a week, and a reviewer reading the
stale version would be reading a description of an experiment that was never run. These tests
make the drift a test failure instead.
"""

from __future__ import annotations

import re

import pytest

from src.common.paths import REPO_ROOT
from src.environment.observation import build_observation_spec
from src.environment.reward import WEIGHT_GRID

DOC = REPO_ROOT / "docs" / "rl_environment.md"


@pytest.fixture(scope="module")
def doc_text() -> str:
    if not DOC.exists():
        pytest.skip("docs/rl_environment.md not present in this checkout")
    return DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def real_spec():
    """The spec for the real 24-station line, which is what the document tabulates."""
    return build_observation_spec(
        station_ids=tuple(f"S{i:02d}" for i in range(24)),
        n_segments=4,
        include_availability_channel=True,
    )


class TestObservationTable:
    def test_every_block_is_documented(self, doc_text: str, real_spec) -> None:
        for block in real_spec.blocks:
            assert f"`{block.name}`" in doc_text, f"{block.name} missing from {DOC.name}"

    def test_no_block_is_documented_that_does_not_exist(self, doc_text: str, real_spec) -> None:
        """Catches a block that was renamed or removed in code but left in the document."""
        documented = set(re.findall(r"^\| `([a-z_0-9]+)` \| \d+ \| \d+:\d+", doc_text, re.M))
        actual = {block.name for block in real_spec.blocks}
        assert documented <= actual, documented - actual

    def test_block_sizes_and_offsets_match(self, doc_text: str, real_spec) -> None:
        rows = dict(
            (name, (int(size), int(start), int(stop)))
            for name, size, start, stop in re.findall(
                r"^\| `([a-z_0-9]+)` \| (\d+) \| (\d+):(\d+)", doc_text, re.M
            )
        )
        offset = 0
        for block in real_spec.blocks:
            assert block.name in rows, block.name
            size, start, stop = rows[block.name]
            assert (size, start, stop) == (block.size, offset, offset + block.size), block.name
            offset += block.size

    def test_documented_totals_match(self, doc_text: str, real_spec) -> None:
        without = build_observation_spec(
            station_ids=tuple(f"S{i:02d}" for i in range(24)),
            n_segments=4,
            include_availability_channel=False,
        )
        assert f"**{real_spec.size} values**" in doc_text
        assert f"**{without.size}** with it" in doc_text

    def test_uncertainty_channels_are_marked(self, doc_text: str, real_spec) -> None:
        for block in real_spec.blocks:
            if not block.carries_uncertainty:
                continue
            row = re.search(rf"^\| `{block.name}` \|.*$", doc_text, re.M)
            assert row and "**yes**" in row.group(0), block.name


class TestRewardTable:
    def test_every_weight_row_is_documented(self, doc_text: str) -> None:
        for weight_id, row in WEIGHT_GRID.items():
            pattern = (
                rf"\| `{weight_id}` \| {row.alpha} \| {row.beta} \| "
                rf"{row.gamma} \| {row.delta} \|"
            )
            assert re.search(pattern, doc_text), weight_id

    def test_no_undocumented_weight_row_exists(self, doc_text: str) -> None:
        documented = set(re.findall(r"^\| `(w_[a-z_]+)` \| [\d.]+ \| ", doc_text, re.M))
        assert documented == set(WEIGHT_GRID)

    def test_action_space_size_matches_the_default_config(self, doc_text: str) -> None:
        from src.environment.metro_env import EnvConfig

        assert f"`Discrete({EnvConfig().k_max + 1})`" in doc_text


class TestProvisionalStatusIsDeclared:
    """Two artefacts are knowingly provisional. The document must say so, because a reader who
    assumes they are final would misread every number produced with them."""

    def test_scaler_status_is_declared(self, doc_text: str) -> None:
        assert "The scaler is not yet fitted" in doc_text

    def test_reward_scale_status_is_declared(self, doc_text: str) -> None:
        assert "provisional" in doc_text.lower()

    def test_committed_reward_scales_are_flagged_provisional(self) -> None:
        import json

        path = REPO_ROOT / "configs" / "reward_scales.json"
        if not path.exists():
            pytest.skip("reward scales not fitted in this checkout")
        provenance = json.loads(path.read_text(encoding="utf-8"))["provenance"]
        if provenance["source_policy"] != "fixed_timetable":
            assert provenance["provisional"] is True

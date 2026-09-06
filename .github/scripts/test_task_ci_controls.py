#!/usr/bin/env python3
"""Minimal unit tests for the two ported task_ci behaviors.

Scope is deliberately limited to what the ports from b37c92a (Harbor 0.22+
reward_stats parsing in result_reward) and b739389 (apply_after=oracle flat
/solution layout in prepare_case) require:

- result_reward: reward_stats path, legacy metrics fallback, and fail-closed on
  malformed / non-numeric / duplicate-trial-id / count-mismatch inputs;
- prepare_case: apply_after=base (default) and apply_after=oracle, the flat
  /solution layout, and reserved-name collision.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("task_ci", HERE / "task_ci.py")
assert SPEC is not None and SPEC.loader is not None
task_ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(task_ci)


def write_result(root: Path, stats: dict) -> Path:
    path = root / "result.json"
    path.write_text(json.dumps({"stats": stats}))
    return path


class ResultRewardTest(unittest.TestCase):
    """b37c92a: Harbor 0.22+ reward_stats with legacy fallback, fail-closed."""

    def test_reward_stats_single_trial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_result(
                Path(tmp),
                {
                    "n_completed_trials": 1,
                    "n_errored_trials": 0,
                    "evals": {"nop__adhoc": {"reward_stats": {"reward": {"1.0": ["t1"]}}}},
                },
            )
            self.assertEqual(task_ci.result_reward(path), (1, 0, [1.0]))

    def test_reward_stats_multiple_trials_expand(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_result(
                Path(tmp),
                {
                    "n_completed_trials": 3,
                    "n_errored_trials": 0,
                    "evals": {
                        "e": {"reward_stats": {"reward": {"1.0": ["a", "b"], "0.0": ["c"]}}}
                    },
                },
            )
            completed, errored, rewards = task_ci.result_reward(path)
            self.assertEqual((completed, errored), (3, 0))
            self.assertEqual(sorted(rewards), [0.0, 1.0, 1.0])

    def test_legacy_metrics_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_result(
                Path(tmp),
                {
                    "n_completed_trials": 1,
                    "n_errored_trials": 0,
                    "evals": {"e": {"metrics": [{"reward": 1.0}]}},
                },
            )
            self.assertEqual(task_ci.result_reward(path), (1, 0, [1.0]))

    def test_reward_stats_preferred_over_legacy_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_result(
                Path(tmp),
                {
                    "n_completed_trials": 1,
                    "n_errored_trials": 0,
                    "evals": {
                        "e": {
                            "reward_stats": {"reward": {"1.0": ["t1"]}},
                            "metrics": [{"reward": 0.0}],
                        }
                    },
                },
            )
            self.assertEqual(task_ci.result_reward(path), (1, 0, [1.0]))

    def test_malformed_reward_map_not_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_result(
                Path(tmp),
                {
                    "n_completed_trials": 1,
                    "n_errored_trials": 0,
                    "evals": {"e": {"reward_stats": {"reward": [["1.0", ["t1"]]]}}},
                },
            )
            with self.assertRaises(task_ci.ContractError):
                task_ci.result_reward(path)

    def test_non_numeric_reward_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_result(
                Path(tmp),
                {
                    "n_completed_trials": 1,
                    "n_errored_trials": 0,
                    "evals": {"e": {"reward_stats": {"reward": {"pass": ["t1"]}}}},
                },
            )
            with self.assertRaises(task_ci.ContractError):
                task_ci.result_reward(path)

    def test_duplicate_trial_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_result(
                Path(tmp),
                {
                    "n_completed_trials": 2,
                    "n_errored_trials": 0,
                    "evals": {
                        "a": {"reward_stats": {"reward": {"1.0": ["dup"]}}},
                        "b": {"reward_stats": {"reward": {"0.0": ["dup"]}}},
                    },
                },
            )
            with self.assertRaises(task_ci.ContractError):
                task_ci.result_reward(path)

    def test_trial_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_result(
                Path(tmp),
                {
                    "n_completed_trials": 2,
                    "n_errored_trials": 0,
                    "evals": {"e": {"reward_stats": {"reward": {"1.0": ["t1"]}}}},
                },
            )
            with self.assertRaises(task_ci.ContractError):
                task_ci.result_reward(path)

    def test_non_integer_trial_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_result(
                Path(tmp),
                {
                    "n_completed_trials": None,
                    "n_errored_trials": 0,
                    "evals": {"e": {"reward_stats": {"reward": {"1.0": ["t1"]}}}},
                },
            )
            with self.assertRaises(task_ci.ContractError):
                task_ci.result_reward(path)


class ControlPreparationTest(unittest.TestCase):
    """b739389: apply_after=base default and apply_after=oracle flat layout."""

    _OMIT = object()

    def make_task(
        self,
        root: Path,
        *,
        apply_after: object = _OMIT,
        extra_oracle_files: tuple[str, ...] = (),
    ) -> Path:
        task = root / "example"
        (task / "solution").mkdir(parents=True)
        (task / "validation").mkdir()
        (task / "solution" / "solve.sh").write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\ngit apply /solution/fix.patch\n"
        )
        (task / "solution" / "fix.patch").write_text("oracle patch\n")
        for name in extra_oracle_files:
            (task / "solution" / name).write_text("reserved\n")
        (task / "task.toml").write_text(
            '[environment]\nworkdir = "/workspace/repo"\n'
        )
        patch = task / "validation" / "control.patch"
        patch.write_text("control patch\n")
        case = {
            "name": "control",
            "patch": patch.name,
            "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
            "expected_reward": 0,
        }
        # Only record apply_after when explicitly provided, so the default test
        # exercises a manifest that omits the key entirely.
        if apply_after is not self._OMIT:
            case["apply_after"] = apply_after
        manifest = {
            "schema_version": "ai_infra_bench_validation_cases.v1",
            "cases": [case],
        }
        (task / "validation" / "ci-cases.json").write_text(json.dumps(manifest))
        return task

    def test_apply_after_omitted_defaults_to_base(self) -> None:
        # ci-cases.json omits apply_after entirely; behavior must equal base.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = self.make_task(root)  # apply_after key absent
            self.assertNotIn(
                "apply_after",
                json.loads((task / "validation" / "ci-cases.json").read_text())[
                    "cases"
                ][0],
            )
            out = root / "prepared"
            agent = task_ci.prepare_case(task, "img:tag", "control", out)
            self.assertEqual(agent, "oracle")
            solve = (out / "solution" / "solve.sh").read_text()
            # base layout applies only the control patch; no oracle entrypoint.
            self.assertIn("git apply /solution/ci-case.patch", solve)
            self.assertNotIn("oracle-solve.sh", solve)
            self.assertFalse((out / "solution" / "oracle-solve.sh").exists())
            self.assertFalse((out / "solution" / "fix.patch").exists())

    def test_invalid_apply_after_rejected_by_manifest(self) -> None:
        # A garbled value must fail loud at manifest validation, never silently
        # fall back to base.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = self.make_task(root, apply_after="orcale")
            with self.assertRaises(task_ci.ContractError):
                task_ci.validation_manifest(task)

    def test_apply_after_oracle_flat_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = self.make_task(root, apply_after="oracle")
            out = root / "prepared"
            task_ci.prepare_case(task, "img:tag", "control", out)
            solution = out / "solution"
            # Oracle files land flat at /solution; entrypoint renamed.
            self.assertTrue((solution / "oracle-solve.sh").exists())
            self.assertTrue((solution / "fix.patch").exists())
            self.assertTrue((solution / "ci-case.patch").exists())
            self.assertFalse((solution / "solve.sh").parent.joinpath("oracle").exists())
            solve = (solution / "solve.sh").read_text()
            # Oracle runs first (its absolute /solution/fix.patch resolves), then
            # the control patch applies.
            self.assertLess(
                solve.index("bash /solution/oracle-solve.sh"),
                solve.index("git apply /solution/ci-case.patch"),
            )
            # The renamed entrypoint still references the flat absolute path.
            self.assertIn(
                "git apply /solution/fix.patch",
                (solution / "oracle-solve.sh").read_text(),
            )

    def test_apply_after_oracle_reserved_name_collision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = self.make_task(
                root, apply_after="oracle", extra_oracle_files=("ci-case.patch",)
            )
            out = root / "prepared"
            with self.assertRaises(task_ci.ContractError):
                task_ci.prepare_case(task, "img:tag", "control", out)


if __name__ == "__main__":
    unittest.main()

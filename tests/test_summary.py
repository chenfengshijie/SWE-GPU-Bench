import argparse
import json
from pathlib import Path

from swe_gpu_bench.evaluate import summary


def write_result(case_dir: Path, payload: dict) -> None:
    case_dir.mkdir(parents=True)
    (case_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")


def test_collect_rows_reads_result_payloads_and_marks_missing(tmp_path: Path) -> None:
    root = tmp_path / "agent_eval"
    write_result(
        root / "org__repo-101",
        {
            "instance_id": "org__repo-101",
            "status": "passed",
            "evaluation_stats": {"failed_commands": 0},
            "agent_stats": {"elapsed_seconds": 12.5},
        },
    )
    (root / "org__repo-102").mkdir()
    (root / "_repo_cache").mkdir()

    rows = summary.collect_rows(root)

    assert [row["instance_id"] for row in rows] == ["org__repo-101", "org__repo-102"]
    assert rows[0]["group"] == "agent_eval"
    assert rows[0]["status"] == "passed"
    assert rows[0]["failed_commands"] == 0
    assert rows[0]["elapsed_seconds"] == 12.5
    assert rows[1]["status"] == "missing"
    assert rows[1]["failed_commands"] is None


def test_collect_rows_reads_perf_fields_per_group(tmp_path: Path) -> None:
    write_result(
        tmp_path / "model_a" / "case-1",
        {
            "status": "passed",
            "perf_status": "improved",
            "elapsed_seconds": 3.0,
            "llm_judgment": {"verdict": "improved", "delta_pct": 12.0},
        },
    )
    write_result(tmp_path / "model_b" / "case-1", {"status": "failed", "perf_status": "regressed"})

    rows = summary.collect_rows(tmp_path, ["model_a", "model_b", "missing_group"])

    assert [(row["group"], row["perf_status"]) for row in rows] == [
        ("model_a", "improved"),
        ("model_b", "regressed"),
    ]
    assert rows[0]["llm_verdict"] == "improved"
    assert rows[0]["delta_pct"] == 12.0
    assert rows[0]["elapsed_seconds"] == 3.0


def test_build_summary_counts_statuses_per_group() -> None:
    def row(group: str, status: str, perf_status=None) -> dict:
        return {**dict.fromkeys(summary.ROW_FIELDS), "group": group, "status": status, "perf_status": perf_status}

    result = summary.build_summary(
        [
            row("a", "passed"),
            row("a", "failed"),
            row("a", "error"),
            row("a", "missing"),
            row("b", "passed", "improved"),
            row("b", "timeout", "unchanged"),
        ]
    )

    assert result["groups"]["a"] == {
        "total": 4,
        "passed": 1,
        "failed": 1,
        "error": 1,
        "missing": 1,
        "other": 0,
        "pass_rate": 0.25,
    }
    assert result["groups"]["b"]["other"] == 1
    assert result["groups"]["b"]["perf_statuses"] == {"improved": 1, "unchanged": 1}
    assert result["groups"]["b"]["improvement_rate"] == 0.5
    assert len(result["rows"]) == 6


def test_run_writes_json_and_csv(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    write_result(root / "case-1", {"status": "passed"})
    parser = argparse.ArgumentParser()
    summary.add_arguments(parser)
    args = parser.parse_args(
        [
            "--eval-root",
            str(root),
            "--json-output",
            str(tmp_path / "summary.json"),
            "--csv-output",
            str(tmp_path / "summary.csv"),
        ]
    )

    assert summary.run(args) == 0
    assert json.loads((tmp_path / "summary.json").read_text())["groups"]["eval"]["passed"] == 1
    assert (tmp_path / "summary.csv").read_text().splitlines()[0].startswith("group,instance_id,status")

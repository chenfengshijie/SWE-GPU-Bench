import json
from pathlib import Path

from swe_gpu_bench.cli import main
from swe_gpu_bench.core.io import load_jsonl, load_verified_records
from swe_gpu_bench.generate import dataset

REPO = "deepspeedai/DeepSpeed"
SLUG = "deepspeedai__DeepSpeed"
PATCH = (
    "diff --git a/deepspeed/runtime/zero.py b/deepspeed/runtime/zero.py\n"
    "--- a/deepspeed/runtime/zero.py\n"
    "+++ b/deepspeed/runtime/zero.py\n"
    "@@ -1 +1 @@\n"
    "-x = 1\n"
    "+x = 2\n"
    "diff --git a/tests/unit/test_zero.py b/tests/unit/test_zero.py\n"
    "--- a/tests/unit/test_zero.py\n"
    "+++ b/tests/unit/test_zero.py\n"
    "@@ -1 +1 @@\n"
    "-assert 1\n"
    "+assert 2\n"
)
PR_META = {
    "repo": REPO,
    "number": 7886,
    "title": "Fix hook count performance regression",
    "body": "body",
    "html_url": "https://github.com/deepspeedai/DeepSpeed/pull/7886",
    "base_ref": "master",
    "head_ref": "branch",
    "base_sha": "base123",
    "commit_messages": ["msg"],
}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_verified_records_dedupes_and_infers_dataset(tmp_path: Path) -> None:
    verified = {"repo": REPO, "pr_number": 7886, "verification_status": "verified"}
    write_json(tmp_path / "gpu_set" / SLUG / "7886.json", verified)
    write_json(tmp_path / "other_set" / SLUG / "7886.json", verified)
    write_json(tmp_path / SLUG / "1.json", {**verified, "pr_number": 1})
    write_json(tmp_path / SLUG / "2.json", {**verified, "pr_number": 2, "verification_status": "failed"})
    write_json(tmp_path / "DeepSpeed-pr3" / "artifacts" / "result.json", {**verified, "pr_number": 3})

    records = load_verified_records(tmp_path)

    assert sorted((record["pr_number"], record.get("dataset")) for record in records) == [
        (1, None),
        (3, None),
        (7886, "gpu_set"),
    ]


def test_task_type_for_maps_pr_category() -> None:
    def scored(primary):
        return {"gpt_result": {"pr_category": {"primary": primary}}}

    assert dataset.task_type_for(scored("bugfix")) == "bug"
    assert dataset.task_type_for(scored("perf")) == "perf"
    assert dataset.task_type_for(scored("refactor")) == "feature"
    assert dataset.task_type_for(None) is None
    assert dataset.task_type_for({"gpt_result": {}}) is None


def test_dataset_build_joins_verified_queries_patches_and_task_type(tmp_path: Path) -> None:
    write_json(
        tmp_path / "verified" / "gpu_set" / SLUG / "7886.json",
        {
            "repo": REPO,
            "pr_number": 7886,
            "head_sha": "head456",
            "verification_status": "verified",
            "setup_commands": ["setup"],
            "test_commands": ["pytest tests/unit/test_zero.py"],
        },
    )
    write_json(
        tmp_path / "verified" / "gpu_set" / SLUG / "1.json",
        {"repo": REPO, "pr_number": 1, "verification_status": "verified"},
    )
    write_json(tmp_path / "queries" / SLUG / "7886.json", {"query": "Fix the hook count."})
    write_json(tmp_path / "prs" / SLUG / "7886.json", PR_META)
    split_dir = tmp_path / "split_patch" / SLUG
    split_dir.mkdir(parents=True)
    (split_dir / "7886.diff").write_text("patch", encoding="utf-8")
    (split_dir / "7886.test.diff").write_text("test patch", encoding="utf-8")
    scored_path = tmp_path / "fused_scored" / SLUG / "fused_scored.jsonl"
    scored_path.parent.mkdir(parents=True)
    scored_path.write_text(
        json.dumps({"number": 7886, "gpt_result": {"pr_category": {"primary": "bugfix"}}}) + "\n", encoding="utf-8"
    )

    rc = main(
        [
            "generate",
            "dataset-build",
            "--verified-dir", str(tmp_path / "verified"),
            "--queries-dir", str(tmp_path / "queries"),
            "--prs-dir", str(tmp_path / "prs"),
            "--split-patch-dir", str(tmp_path / "split_patch"),
            "--scored-dir", str(tmp_path / "fused_scored"),
            "--output-dir", str(tmp_path / "out"),
        ]
    )

    assert rc == 0
    rows = load_jsonl(tmp_path / "out" / "swe_gpu_bench.jsonl")
    assert len(rows) == 1
    row = rows[0]
    assert row["instance_id"] == "deepspeedai__DeepSpeed__7886"
    assert row["dataset"] == "gpu_set"
    assert (row["base_commit"], row["head_commit"]) == ("base123", "head456")
    assert (row["patch"], row["test_patch"]) == ("patch", "test patch")
    assert row["problem_statement"] == "Fix the hook count."
    assert row["task_type"] == "bug"
    assert load_jsonl(tmp_path / "out" / "gpu_set.jsonl") == rows
    report = json.loads((tmp_path / "out" / "swe_gpu_bench.check.json").read_text(encoding="utf-8"))
    assert report["rows"] == 1
    assert "missing_commit" in report["issue_counts"]


def test_build_perf_instance_carries_perf_command_fields() -> None:
    instance = dataset.build_perf_instance(
        perf_result={
            "repo": REPO,
            "pr_number": 7886,
            "base_commit": "base123",
            "head_commit": "head456",
            "status": "ready",
            "setup_commands": ["setup"],
            "test_commands": ["test"],
            "perf_setup_commands": ["perf-setup"],
            "perf_commands": ["perf"],
            "perf_context": {"perf_kind": "latency", "measurement_hint": "latency_ms"},
        },
        pr_meta=PR_META,
        patch="patch",
        test_patch="test patch",
    )

    assert instance["instance_id"] == "deepspeedai__DeepSpeed__7886"
    assert instance["task_type"] == "perf"
    assert instance["perf_setup_commands"] == ["perf-setup"]
    assert instance["perf_commands"] == ["perf"]
    assert instance["perf_context"]["perf_kind"] == "latency"


def test_perf_dataset_build_only_emits_ready_cases_and_splits_patch(tmp_path: Path) -> None:
    ready = {
        "repo": REPO,
        "pr_number": 7886,
        "base_commit": "base123",
        "head_commit": "head456",
        "status": "ready",
        "setup_commands": ["setup"],
        "test_commands": ["pytest tests/unit/test_zero.py"],
        "perf_setup_commands": ["perf-setup"],
        "perf_commands": ["perf"],
        "perf_context": {"perf_kind": "latency"},
    }
    write_json(tmp_path / "cases" / "DeepSpeed-pr7886" / "logs" / "result.json", ready)
    write_json(
        tmp_path / "cases" / "DeepSpeed-pr3748" / "logs" / "result.json",
        {**ready, "pr_number": 3748, "status": "blocked"},
    )
    write_json(tmp_path / "prs" / SLUG / "7886.json", PR_META)
    patch_file = tmp_path / "patches" / SLUG / "7886.diff"
    patch_file.parent.mkdir(parents=True)
    patch_file.write_text(PATCH, encoding="utf-8")

    rc = main(
        [
            "generate",
            "perf-dataset-build",
            "--case-root", str(tmp_path / "cases"),
            "--prs-dir", str(tmp_path / "prs"),
            "--patches-dir", str(tmp_path / "patches"),
            "--output-dir", str(tmp_path / "out"),
        ]
    )

    assert rc == 0
    rows = load_jsonl(tmp_path / "out" / "swe_gpu_bench_perf.jsonl")
    assert [row["pr_number"] for row in rows] == [7886]
    assert "deepspeed/runtime/zero.py" in rows[0]["patch"]
    assert "tests/unit/test_zero.py" not in rows[0]["patch"]
    assert "tests/unit/test_zero.py" in rows[0]["test_patch"]


def test_export_projects_rows_onto_published_schema(tmp_path: Path) -> None:
    source = tmp_path / "dataset.jsonl"
    rows = [
        {
            "instance_id": "a__b__1",
            "dataset": "gpu_set",
            "repo": "a/b",
            "pr_number": 1,
            "pr_title": "internal only",
            "setup_commands": None,
            "test_commands": ["pytest"],
            "verification_status": "verified",
        },
        {"instance_id": "a__b__2", "repo": "a/b", "pr_number": 2, "docker_image": "custom:img"},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("a__b__2\n", encoding="utf-8")

    assert main(["generate", "export", "--dataset", str(source), "--output-dir", str(tmp_path / "out"),
                 "--docker-image", "verlai/verl:latest", "--keep-invalid"]) == 0
    assert main(["generate", "export", "--dataset", str(source), "--output-dir", str(tmp_path / "out"),
                 "--split", "subset", "--instance-id-file", str(ids_file), "--keep-invalid"]) == 0

    exported = load_jsonl(tmp_path / "out" / "test.jsonl")
    assert [list(row) for row in exported] == [list(dataset.EXPORT_FIELDS)] * 2
    assert exported[0]["setup_commands"] == []
    assert exported[0]["docker_image"] == "verlai/verl:latest"
    assert exported[1]["docker_image"] == "custom:img"
    subset = load_jsonl(tmp_path / "out" / "subset.jsonl")
    assert [row["instance_id"] for row in subset] == ["a__b__2"]


def valid_row(instance_id: str, pr_number: int, **overrides) -> dict:
    return {
        "instance_id": instance_id,
        "repo": REPO,
        "pr_number": pr_number,
        "base_commit": "b" * 40,
        "merged_at": "2024-02-02T00:00:00Z",
        "problem_statement": "Fix the hook count.",
        "patch": PATCH.split("diff --git a/tests/")[0].replace("x = ", "torch.cuda.x = "),
        "test_patch": "diff --git a/tests/" + PATCH.split("diff --git a/tests/")[1],
        "test_commands": ["pytest tests/unit/test_zero.py"],
        "verification_status": "verified",
        **overrides,
    }


def test_export_drops_rows_with_check_errors_and_applies_verify_results(tmp_path: Path) -> None:
    source = tmp_path / "dataset.jsonl"
    rows = [
        valid_row("ds__1", 1, verification_status="reproduced"),
        valid_row("ds__2", 2, merged_at=None, patch=valid_row("x", 0)["patch"].replace("2", "3")),
        valid_row("ds__3", 3, verification_status="reproduced", patch=valid_row("x", 0)["patch"].replace("2", "4")),
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    write_json(tmp_path / "verify" / "ds__1" / "result.json", {"instance_id": "ds__1", "verification_status": "verified"})
    write_json(tmp_path / "verify" / "ds__2" / "result.json", {"instance_id": "ds__2", "verification_status": "verified"})

    assert main(["generate", "export", "--dataset", str(source), "--output-dir", str(tmp_path / "out"),
                 "--verify-root", str(tmp_path / "verify")]) == 0

    exported = load_jsonl(tmp_path / "out" / "test.jsonl")
    assert [(row["instance_id"], row["verification_status"]) for row in exported] == [("ds__1", "verified")]
    report = json.loads((tmp_path / "out" / "test.check.json").read_text(encoding="utf-8"))
    assert {entry["instance_id"]: [error["code"] for error in entry["errors"]] for entry in report["instances"]} == {
        "ds__2": ["unmerged_pr"],
        "ds__3": ["unverified"],
    }


def test_export_record_keeps_perf_commands_when_present() -> None:
    row = dataset.export_record({"instance_id": "p__1", "perf_commands": ["bench"], "perf_setup_commands": []}, None)

    assert row["perf_commands"] == ["bench"]
    assert "perf_setup_commands" not in row


def test_dataset_finalize_resplits_patch_with_test_patch_and_keeps_fields(tmp_path: Path) -> None:
    leaked = (
        "diff --git a/include/cutlass/sm90_warpspecialized.hpp b/include/cutlass/sm90_warpspecialized.hpp\n"
        "--- a/include/cutlass/sm90_warpspecialized.hpp\n"
        "+++ b/include/cutlass/sm90_warpspecialized.hpp\n"
        "@@ -1 +1 @@\n"
        "+struct Collective {};\n"
    )
    entry = valid_row("ds__1", 1, docker_image="custom:img")
    entry["test_patch"] += leaked
    source = tmp_path / "in.jsonl"
    source.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    assert main(["generate", "dataset-finalize", "--input", str(source), "--output-dir", str(tmp_path / "out"),
                 "--prompt-file", str(tmp_path / "missing.md")]) == 0

    [row] = load_jsonl(tmp_path / "out" / "swe_gpu_bench.jsonl")
    assert "sm90_warpspecialized.hpp" in row["patch"]
    assert "sm90_warpspecialized.hpp" not in row["test_patch"]
    assert "tests/unit/test_zero.py" in row["test_patch"]
    assert row["docker_image"] == "custom:img"
    assert row["merged_at"] == "2024-02-02T00:00:00Z"
    assert row["problem_statement"] == "Fix the hook count."

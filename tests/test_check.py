import json

from swe_gpu_bench import cli
from swe_gpu_bench.generate import check
from swe_gpu_bench.generate.check import (
    apply_verify_status,
    check_dataset,
    check_row,
    clean_rows,
    cpu_only_reason,
    installs_unpinned_torch,
)


def section(path, *lines, new=False):
    header = f"diff --git a/{path} b/{path}\n" + ("new file mode 100644\n" if new else "")
    return header + f"--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n" + "\n".join(lines) + "\n"


def make_row(**overrides):
    row = {
        "instance_id": "acme__attn__1",
        "repo": "acme/attn",
        "base_commit": "a" * 40,
        "problem_statement": "Attention output is wrong for odd widths.",
        "patch": section("csrc/attn.cu", "-int width = 1;", "+int width = 2;", "+int height = 3;"),
        "test_patch": section("tests/test_attn.py", "+def test_attn():", "+    assert run_attn()"),
        "setup_commands": ["pip install -e ."],
        "test_commands": ["pytest tests/test_attn.py"],
        "verification_status": "verified",
    }
    row.update(overrides)
    return row


def codes(issues):
    return {issue.code for issue in issues}


def test_check_row_accepts_a_clean_row():
    assert check_row(make_row(), "verified") == []


def test_check_row_reports_rows_that_cannot_be_benchmark_tasks():
    leaked_header = section("include/cutlass/gemm/sm90_mma_warpspecialized.hpp", "+struct Collective {};")
    row = make_row(
        base_commit=None,
        merged_at=None,
        patch=make_row()["patch"] + "diff --git a/img.png b/img.png\nBinary files a/img.png and b/img.png differ\n",
        test_patch=make_row()["test_patch"] + leaked_header,
        setup_commands=["/home/example-user/.local/bin/python3.10 -m pip install -e ."],
        test_commands=["JAX_PLATFORMS=cpu pytest tests/test_attn.py"],
    )

    issues = check_row(row, "reproduced")

    assert {
        "missing_commit",
        "unverified",
        "unmerged_pr",
        "split_mismatch",
        "unappliable_binary",
        "cpu_only_tests",
        "host_path",
    } <= codes(issues)
    mismatch = next(issue for issue in issues if issue.code == "split_mismatch")
    assert mismatch.detail == "test_patch->patch: include/cutlass/gemm/sm90_mma_warpspecialized.hpp"
    assert check.has_error(issues)


def test_check_row_reports_warnings_without_errors():
    row = make_row(
        problem_statement="Add a fused attention entry point.",
        patch=section("csrc/attn.cu", "+// sm_100a only", "+int width = 2;")
        + section("python/attn.py", "+def fused_attention_v2(query, key):", "+    return query"),
        test_patch=section("tests/test_attn.py", "+def test_fused():", "+    assert fused_attention_v2(q, k)")
        + section("CMakeLists.txt", "+add_library(attn src/attn_v2.cu)", "+add_test(test_attn tests/test_attn.py)"),
        setup_commands=["pip install torch --index-url https://download.pytorch.org/whl/cu124", "/usr/local/cuda-12.4/bin/nvcc --version"],
        test_commands=["pip install einops && pytest tests/test_attn.py"],
    )

    issues = check_row(row, "verified")

    assert codes(issues) == {
        "cuda_path",
        "network_in_tests",
        "unpinned_torch",
        "blackwell_signal",
        "underspecified_interface",
        "harness_entangled",
    }
    assert not check.has_error(issues)
    assert next(issue for issue in issues if issue.code == "underspecified_interface").detail == "fused_attention_v2"
    assert next(issue for issue in issues if issue.code == "harness_entangled").detail == "CMakeLists.txt"


def test_check_row_warns_about_missing_tests_size_and_gpu_signal(monkeypatch):
    monkeypatch.setattr(check, "MAX_CODE_LINES", 1)
    row = make_row(patch=section("src/tokenizer.py", "-x = 1", "+x = 2"), test_patch="")

    assert codes(check_row(row, "verified")) == {"no_test_patch", "large_patch", "no_gpu_signal"}


def test_cpu_only_reason_spots_cpu_forcing_commands():
    assert cpu_only_reason("CUDA_VISIBLE_DEVICES= pytest tests") == "CUDA_VISIBLE_DEVICES="
    assert cpu_only_reason("NO_CUDA=1 python setup.py develop") == "NO_CUDA=1"
    assert cpu_only_reason('TVM_TEST_TARGETS="llvm" pytest tests') == 'TVM_TEST_TARGETS="llvm"'
    assert cpu_only_reason("bazel test //xla/service:spmd_test") == "bazel test without --config=cuda"
    assert cpu_only_reason("CUDA_VISIBLE_DEVICES=0 pytest tests") is None
    assert cpu_only_reason('TVM_TEST_TARGETS="llvm;cuda" pytest tests') is None
    assert cpu_only_reason("bazel test --config=cuda //xla/service/gpu:gemm_test") is None


def test_installs_unpinned_torch_requires_an_exact_pin():
    assert installs_unpinned_torch("pip install torch")
    assert installs_unpinned_torch("uv pip install 'torch>=2.1' numpy")
    assert not installs_unpinned_torch("pip install torch==2.4.0")
    assert not installs_unpinned_torch("pip install torchvision einops")


def test_check_dataset_flags_repeated_ids_and_mirrored_changes():
    original = make_row()
    mirrored = make_row(
        instance_id="mirror__attn__9",
        patch=original["patch"].replace("csrc/attn.cu", "third_party/attn/csrc/attn.cu"),
        test_patch=original["test_patch"].replace("tests/test_attn.py", "third_party/attn/tests/test_attn.py"),
        test_commands=["pytest third_party/attn/tests/test_attn.py"],
    )
    repeated = make_row(patch=section("csrc/other.cu", "+int a;", "+int b;", "+int c;"))

    reports = check_dataset([original, mirrored, repeated])

    assert reports[0] == []
    assert [issue.detail for issue in reports[1]] == ["same change as acme__attn__1"]
    assert [issue.detail for issue in reports[2]] == ["instance_id appears earlier in the dataset"]


def test_check_dataset_keeps_the_first_usable_copy_of_a_change():
    unusable = make_row(verification_status="reproduced")
    usable = make_row(instance_id="mirror__attn__9")

    reports = check_dataset([unusable, usable, make_row(instance_id="mirror__attn__10", verification_status="reproduced")])

    assert codes(reports[0]) == {"unverified", "duplicate"}
    assert reports[1] == []
    assert [issue.detail for issue in reports[2] if issue.code == "duplicate"] == ["same change as mirror__attn__9"]


def test_check_dataset_reports_duplicates_among_unusable_rows():
    first = make_row(verification_status="reproduced")
    second = make_row(instance_id="mirror__attn__9", verification_status="reproduced")

    reports = check_dataset([first, second])

    assert codes(reports[0]) == {"unverified"}
    assert [issue.detail for issue in reports[1] if issue.code == "duplicate"] == ["same change as acme__attn__1"]


def test_verify_results_replace_the_recorded_status():
    rows = [make_row(verification_status="reproduced"), make_row(instance_id="acme__attn__2", patch=section("csrc/b.cu", "+int a;", "+int b;", "+int c;"))]
    verify_results = {"acme__attn__1": {"verification_status": "verified"}}

    reports = check_dataset(rows, verify_results=verify_results)
    kept = apply_verify_status(clean_rows(rows, reports), verify_results)

    assert reports[0] == []
    assert [issue.to_json() for issue in reports[1]] == [{"code": "unverified", "detail": "verification_status=None"}]
    assert [(row["instance_id"], row["verification_status"]) for row in kept] == [("acme__attn__1", "verified")]


def test_check_dataset_skips_ignored_codes():
    reports = check_dataset([make_row(verification_status="reproduced")], ignore=["unverified"])

    assert reports == [[]]


def test_dataset_check_command_writes_report_and_clean_rows(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    output = tmp_path / "clean.jsonl"
    rows = [make_row(), make_row(instance_id="acme__attn__2", base_commit="HEAD", patch=section("csrc/b.cu", "+int a;", "+int b;", "+int c;"))]
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    assert cli.main(["generate", "dataset-check", "--dataset", str(dataset), "--output", str(output)]) == 0

    report = json.loads((tmp_path / "dataset.check.json").read_text(encoding="utf-8"))
    assert report["rows"] == 2
    assert report["rows_with_errors"] == 1
    assert report["issue_counts"] == {"missing_commit": {"severity": "error", "rows": 1}}
    assert report["instances"][0]["errors"] == [{"code": "missing_commit", "detail": "base_commit='HEAD'"}]
    assert [json.loads(line)["instance_id"] for line in output.read_text(encoding="utf-8").splitlines()] == ["acme__attn__1"]

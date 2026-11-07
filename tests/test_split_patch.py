import json
import subprocess

from swe_gpu_bench.core.diff import is_test_like_path, parse_diff_sections
from swe_gpu_bench.generate.split_patch import (
    collect_verified_targets,
    extract_test_signals,
    split_patch_text,
    split_verified_patches,
)


def test_parse_diff_sections_preserves_order():
    diff_text = """diff --git a/src/main.py b/src/main.py
index 111..222 100644
--- a/src/main.py
+++ b/src/main.py
@@ -1 +1 @@
-x = 1
+x = 2
diff --git a/tests/test_main.py b/tests/test_main.py
index 333..444 100644
--- a/tests/test_main.py
+++ b/tests/test_main.py
@@ -0,0 +1,2 @@
+def test_main():
+    assert True
"""
    sections = parse_diff_sections(diff_text)
    assert [section.new_path for section in sections] == ["src/main.py", "tests/test_main.py"]
    assert sections[0].raw_text.startswith("diff --git a/src/main.py")
    assert sections[1].raw_text.startswith("diff --git a/tests/test_main.py")


def test_extract_test_signals_from_multiple_command_shapes():
    signals = extract_test_signals(
        [
            "python -m pytest repo/tests/test_alpha.py::test_one",
            "cd repo && python tests/test_beta.py",
            "python - <<'PY'\nfrom tests.test_gamma import test_case\nPY",
            "cd build && ./test_norm --gtest_list_tests",
            "ctest --test-dir build -R '^ctest_examples_41_fmha_backward_python$' --output-on-failure",
        ]
    )

    assert "tests/test_alpha.py" in signals.test_files
    assert "tests/test_beta.py" in signals.test_files
    assert "tests/test_gamma.py" in signals.test_files
    assert "test_norm" in signals.test_targets
    assert "ctest_examples_41_fmha_backward_python" in signals.test_targets


def test_split_patch_text_keeps_cmake_support_with_direct_test():
    diff_text = """diff --git a/CMakeLists.txt b/CMakeLists.txt
index 111..222 100644
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -1 +1,2 @@
 add_library(core src/core.cpp)
+add_executable(test_norm src/test_norm.cu)
diff --git a/src/core.cu b/src/core.cu
index 333..444 100644
--- a/src/core.cu
+++ b/src/core.cu
@@ -1 +1 @@
-int x = 1;
+int x = 2;
diff --git a/src/test_norm.cu b/src/test_norm.cu
index 555..666 100644
--- a/src/test_norm.cu
+++ b/src/test_norm.cu
@@ -0,0 +1,2 @@
+TEST(Norm, Works) {}
+int main() {}
"""
    signals = extract_test_signals(["cd build && ./test_norm --gtest_color=no"])
    split = split_patch_text(diff_text, signals)

    assert "src/test_norm.cu" in split.test_patch_text
    assert "add_executable(test_norm src/test_norm.cu)" in split.test_patch_text
    assert "src/core.cu" in split.patch_text


def test_split_patch_text_does_not_move_product_code_for_ctest_target():
    diff_text = """diff --git a/CMakeLists.txt b/CMakeLists.txt
index 111..222 100644
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -1 +1,2 @@
 add_executable(app src/app.cpp)
+add_test(NAME ctest_examples_41_fmha_backward_python COMMAND fmha_backward_test.py)
diff --git a/examples/41_fused_multi_head_attention/fmha_backward_test.py b/examples/41_fused_multi_head_attention/fmha_backward_test.py
index 333..444 100644
--- a/examples/41_fused_multi_head_attention/fmha_backward_test.py
+++ b/examples/41_fused_multi_head_attention/fmha_backward_test.py
@@ -1 +1 @@
-assert False
+assert True
diff --git a/examples/41_fused_multi_head_attention/kernel_backward.h b/examples/41_fused_multi_head_attention/kernel_backward.h
index 555..666 100644
--- a/examples/41_fused_multi_head_attention/kernel_backward.h
+++ b/examples/41_fused_multi_head_attention/kernel_backward.h
@@ -1 +1 @@
-int old_value;
+int new_value;
"""
    signals = extract_test_signals(
        ["ctest --test-dir artifacts/build_min2 -R ctest_examples_41_fmha_backward_python --output-on-failure"]
    )
    split = split_patch_text(diff_text, signals)

    assert "fmha_backward_test.py" in split.test_patch_text
    assert "add_test(NAME ctest_examples_41_fmha_backward_python" in split.test_patch_text
    assert "kernel_backward.h" in split.patch_text


def test_split_verified_patches_writes_outputs_and_empty_test_patch(tmp_path):
    verified_path = tmp_path / "verified" / "BenchMarkv1" / "acme__project" / "7.json"
    verified_path.parent.mkdir(parents=True)
    verified_path.write_text(
        json.dumps(
            {
                "repo": "acme/project",
                "pr_number": 7,
                "verification_status": "verified",
                "test_commands": ["python -m pytest repo/tests/test_parser.py::test_parse"],
            }
        ),
        encoding="utf-8",
    )
    patch_path = tmp_path / "patches" / "acme__project" / "7.diff"
    patch_path.parent.mkdir(parents=True)
    patch_path.write_text(
        """diff --git a/src/parser.py b/src/parser.py
index 111..222 100644
--- a/src/parser.py
+++ b/src/parser.py
@@ -1 +1 @@
-x = 1
+x = 2
""",
        encoding="utf-8",
    )

    processed, missing = split_verified_patches(
        tmp_path / "verified",
        tmp_path / "patches",
        tmp_path / "split_patch",
    )

    assert processed == 1
    assert missing == 0
    assert (tmp_path / "split_patch" / "acme__project" / "7.diff").read_text(encoding="utf-8") == patch_path.read_text(encoding="utf-8")
    assert (tmp_path / "split_patch" / "acme__project" / "7.test.diff").read_text(encoding="utf-8") == ""


def test_collect_verified_targets_reads_verify_output_and_skips_unverified(tmp_path):
    verified = tmp_path / "verify" / "project-pr7" / "artifacts" / "result.json"
    failed = tmp_path / "verify" / "project-pr8" / "artifacts" / "result.json"
    sample = tmp_path / "verify" / "project-pr7" / "inputs" / "sample.json"
    for path in (verified, failed, sample):
        path.parent.mkdir(parents=True, exist_ok=True)
    verified.write_text(
        json.dumps(
            {"repo": "acme/project", "pr_number": 7, "verification_status": "verified", "test_commands": ["pytest tests/test_a.py"]}
        ),
        encoding="utf-8",
    )
    failed.write_text(
        json.dumps({"repo": "acme/project", "pr_number": 8, "verification_status": "test_failed", "test_commands": ["x"]}),
        encoding="utf-8",
    )
    sample.write_text(json.dumps({"repo": "acme/project", "number": 7}), encoding="utf-8")

    assert collect_verified_targets(tmp_path / "verify") == {("acme/project", 7): ["pytest tests/test_a.py"]}


def test_split_output_patches_pass_git_apply_check(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    (repo / "src").mkdir()
    (repo / "tests").mkdir()
    (repo / "src" / "main.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "tests" / "test_main.py").write_text("def test_old():\n    assert True\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )

    diff_text = """diff --git a/src/main.py b/src/main.py
index 1c23ec5..b8af43e 100644
--- a/src/main.py
+++ b/src/main.py
@@ -1 +1 @@
-x = 1
+x = 2
diff --git a/tests/test_main.py b/tests/test_main.py
index ec4c0be..68e6d76 100644
--- a/tests/test_main.py
+++ b/tests/test_main.py
@@ -1,2 +1,2 @@
-def test_old():
+def test_new():
     assert True
"""
    split = split_patch_text(diff_text, extract_test_signals(["pytest tests/test_main.py::test_new"]))
    patch_file = tmp_path / "patch.diff"
    test_patch_file = tmp_path / "test_patch.diff"
    patch_file.write_text(split.patch_text, encoding="utf-8")
    test_patch_file.write_text(split.test_patch_text, encoding="utf-8")

    subprocess.run(["git", "apply", "--check", str(patch_file)], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "apply", "--check", str(test_patch_file)], cwd=repo, check=True, capture_output=True, text=True)


def test_is_test_like_path_matches_existing_hint_behavior():
    assert is_test_like_path("tests/test_main.py") is True
    assert is_test_like_path("src/main.py") is False


def test_is_test_like_path_requires_whole_test_tokens():
    for path in (
        "test/unit/gemm/device/gemm_testbed.h",
        "csrc/flash_attn_test.cu",
        "python/test_ops.py",
        "torch/testing/_internal/common_cuda.py",
        "xla/service/gpu/FusionTest.cpp",
        "conftest.py",
        "scripts/task_jit_run_tests_part4.sh",
    ):
        assert is_test_like_path(path) is True, path
    for path in (
        "include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp",
        "cub/cub/block/specializations/block_topk_air.cuh",
        "vllm/spec_decode/metrics.py",
        "tools/latest_version.py",
        "src/contest.cu",
    ):
        assert is_test_like_path(path) is False, path


def test_split_patch_text_judges_build_files_by_changed_lines_only():
    diff_text = """diff --git a/CMakeLists.txt b/CMakeLists.txt
index 111..222 100644
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -1,2 +1,3 @@
 add_subdirectory(tests)
 add_library(core src/core.cu)
+target_sources(core PRIVATE src/fast_path.cu)
diff --git a/tests/test_core.py b/tests/test_core.py
index 333..444 100644
--- a/tests/test_core.py
+++ b/tests/test_core.py
@@ -1 +1 @@
-assert False
+assert True
"""
    split = split_patch_text(diff_text, extract_test_signals(["pytest tests/test_core.py"]))

    assert "CMakeLists.txt" in split.patch_text
    assert "tests/test_core.py" in split.test_patch_text


def test_split_patch_text_hides_executed_example_program_and_its_new_helpers():
    diff_text = """diff --git a/include/lib/gemm.hpp b/include/lib/gemm.hpp
index 111..222 100644
--- a/include/lib/gemm.hpp
+++ b/include/lib/gemm.hpp
@@ -1 +1 @@
-int scale_mode = 0;
+int scale_mode = 1;
diff --git a/examples/65_blockwise/65_blockwise.cu b/examples/65_blockwise/65_blockwise.cu
new file mode 100644
index 0000000..333
--- /dev/null
+++ b/examples/65_blockwise/65_blockwise.cu
@@ -0,0 +1 @@
+#include "reference/host/blockwise.h"
diff --git a/examples/65_blockwise/reference/host/blockwise.h b/examples/65_blockwise/reference/host/blockwise.h
new file mode 100644
index 0000000..444
--- /dev/null
+++ b/examples/65_blockwise/reference/host/blockwise.h
@@ -0,0 +1 @@
+void reference_gemm();
diff --git a/examples/41_fmha/kernel_backward.h b/examples/41_fmha/kernel_backward.h
index 555..666 100644
--- a/examples/41_fmha/kernel_backward.h
+++ b/examples/41_fmha/kernel_backward.h
@@ -1 +1 @@
-int old_value;
+int new_value;
"""
    signals = extract_test_signals(
        ["CUDA_VISIBLE_DEVICES=0 artifacts/build/examples/65_blockwise/65_blockwise --m=256 && cd build && python3 -V"]
    )
    split = split_patch_text(diff_text, signals)

    assert signals.executables == frozenset({"65_blockwise"})
    assert "examples/65_blockwise/65_blockwise.cu" in split.test_patch_text
    assert "examples/65_blockwise/reference/host/blockwise.h" in split.test_patch_text
    assert "include/lib/gemm.hpp" in split.patch_text
    assert "examples/41_fmha/kernel_backward.h" in split.patch_text


def test_split_patch_text_hides_example_program_named_by_ctest_target():
    diff_text = """diff --git a/include/lib/mixed_input.hpp b/include/lib/mixed_input.hpp
index 111..222 100644
--- a/include/lib/mixed_input.hpp
+++ b/include/lib/mixed_input.hpp
@@ -1 +1 @@
-int packed = 0;
+int packed = 1;
diff --git a/examples/55_mixed/55_int4_gemm.cu b/examples/55_mixed/55_int4_gemm.cu
new file mode 100644
index 0000000..333
--- /dev/null
+++ b/examples/55_mixed/55_int4_gemm.cu
@@ -0,0 +1 @@
+#include "reorder_utils.hpp"
diff --git a/examples/55_mixed/reorder_utils.hpp b/examples/55_mixed/reorder_utils.hpp
new file mode 100644
index 0000000..444
--- /dev/null
+++ b/examples/55_mixed/reorder_utils.hpp
@@ -0,0 +1 @@
+void reorder();
diff --git a/examples/55_mixed/55_fp8_gemm.cu b/examples/55_mixed/55_fp8_gemm.cu
index 555..666 100644
--- a/examples/55_mixed/55_fp8_gemm.cu
+++ b/examples/55_mixed/55_fp8_gemm.cu
@@ -1 +1 @@
-int scale = 0;
+int scale = 1;
diff --git a/examples/55_mixed/CMakeLists.txt b/examples/55_mixed/CMakeLists.txt
index 777..888 100644
--- a/examples/55_mixed/CMakeLists.txt
+++ b/examples/55_mixed/CMakeLists.txt
@@ -1 +1,2 @@
 cutlass_example_add_executable(55_fp8_gemm 55_fp8_gemm.cu)
+cutlass_example_add_executable(55_int4_gemm 55_int4_gemm.cu TEST_COMMAND_OPTIONS TEST_SMALL)
"""
    signals = extract_test_signals(["ctest --test-dir artifacts/build --output-on-failure -R '^ctest_examples_55_int4_gemm_'"])

    split = split_patch_text(diff_text, signals)

    assert "examples/55_mixed/55_int4_gemm.cu" in split.test_patch_text
    assert "examples/55_mixed/reorder_utils.hpp" in split.test_patch_text
    assert "examples/55_mixed/CMakeLists.txt" in split.test_patch_text
    assert "include/lib/mixed_input.hpp" in split.patch_text
    assert "examples/55_mixed/55_fp8_gemm.cu" in split.patch_text


def test_extract_test_signals_treats_executed_python_scripts_as_test_files():
    signals = extract_test_signals(
        ["cd repo && python examples/blackwell/launch.py --n 4 && python setup.py develop && python -m pytest -q"]
    )

    assert "examples/blackwell/launch.py" in signals.test_files
    assert not any(path.endswith("setup.py") for path in signals.test_files)

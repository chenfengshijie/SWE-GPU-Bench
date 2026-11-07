import json
from collections import Counter

from swe_gpu_bench.collect.filter import (
    FilterCriteria,
    extract_resolved_issues,
    filter_prs,
    has_test_patch,
    rejection_reasons,
    targets_default_branch,
)


GPU_FIX_DIFF = """diff --git a/csrc/parser.cu b/csrc/parser.cu
index 111..222 100644
--- a/csrc/parser.cu
+++ b/csrc/parser.cu
@@ -1 +1 @@
-int width = 1;
+int width = 2;
diff --git a/tests/test_parser.py b/tests/test_parser.py
index 333..444 100644
--- a/tests/test_parser.py
+++ b/tests/test_parser.py
@@ -0,0 +1,2 @@
+def test_parser():
+    assert True
"""


def write_pr(tmp_path, number, diff_text=GPU_FIX_DIFF, **overrides):
    prs_dir = tmp_path / "prs" / "acme__project"
    patches_dir = tmp_path / "patches" / "acme__project"
    prs_dir.mkdir(parents=True, exist_ok=True)
    patches_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "repo": "acme/project",
        "number": number,
        "title": "Fix parser",
        "body": f"Fixes #{number}",
        "html_url": f"https://github.com/acme/project/pull/{number}",
        "commit_messages": [],
        "merged_at": "2024-02-02T00:00:00Z",
        "default_branch": "main",
        "base_ref": "main",
        **overrides,
    }
    (prs_dir / f"{number}.json").write_text(json.dumps(payload), encoding="utf-8")
    if diff_text is not None:
        (patches_dir / f"{number}.diff").write_text(diff_text, encoding="utf-8")


def test_extract_resolved_issues_matches_swebench_style_keywords():
    payload = {
        "title": "fix tokenizer edge case",
        "body": "This closes #12 and resolves acme/project#34",
        "commit_messages": [
            "cleanup",
            "fixed https://github.com/acme/project/issues/56",
        ],
    }

    assert extract_resolved_issues(payload) == ["12", "34", "56"]


def test_has_test_patch_detects_test_files():
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
@@ -0,0 +1,4 @@
+def test_main():
+    assert 1 == 1
"""

    assert has_test_patch(diff_text) is True


def test_filter_prs_requires_resolved_issue_and_test_patch(tmp_path):
    write_pr(tmp_path, 7)

    matches, rejections = filter_prs(tmp_path / "prs", tmp_path / "patches")

    assert len(matches) == 1
    assert matches[0]["resolved_issues"] == ["7"]
    assert matches[0]["number"] == 7
    assert matches[0]["code_files"] == 1
    assert rejections == Counter()


def test_filter_prs_counts_rejection_reasons(tmp_path):
    cpu_only_diff = GPU_FIX_DIFF.replace("csrc/parser.cu", "src/tokenizer.py")
    tests_only_diff = GPU_FIX_DIFF[GPU_FIX_DIFF.index("diff --git a/tests/") :]
    write_pr(tmp_path, 1)
    write_pr(tmp_path, 2, merged_at=None)
    write_pr(tmp_path, 3, base_ref="release/2.1")
    write_pr(tmp_path, 4, cpu_only_diff)
    write_pr(tmp_path, 5, tests_only_diff)
    write_pr(tmp_path, 6, body="Speeds up the parser kernel")
    write_pr(tmp_path, 8, None)

    matches, rejections = filter_prs(tmp_path / "prs", tmp_path / "patches")

    assert [match["number"] for match in matches] == [1]
    assert rejections == Counter(
        {
            "not_merged": 1,
            "non_default_base": 1,
            "no_gpu_signal": 1,
            "no_code_change": 1,
            "no_resolved_issue": 1,
            "missing_patch": 1,
        }
    )

    relaxed = FilterCriteria(require_merged=False, require_default_branch=False, require_issue=False)
    relaxed_matches, _ = filter_prs(tmp_path / "prs", tmp_path / "patches", relaxed)
    assert sorted(match["number"] for match in relaxed_matches) == [1, 2, 3, 6]


def test_targets_default_branch_prefers_recorded_default_then_release_patterns():
    assert targets_default_branch({"base_ref": "main", "default_branch": "main"}) is True
    assert targets_default_branch({"base_ref": "v2.1", "default_branch": "main"}) is False
    assert targets_default_branch({"base_ref": "release/2.1"}) is False
    assert targets_default_branch({"base_ref": "v0.4.x"}) is False
    assert targets_default_branch({"base_ref": "gh/alice/12/base"}) is False
    assert targets_default_branch({"base_ref": "master"}) is True
    assert targets_default_branch({}) is None


def test_rejection_reasons_flags_oversized_and_binary_diffs():
    signals = {
        "has_patch": True,
        "resolves_issues": True,
        "has_test_patch": True,
        "merged": True,
        "targets_default_branch": True,
        "gpu_signal": True,
        "code_files": 31,
        "code_changed_lines": 1001,
        "binary_files": ["docs/plot.png"],
    }

    assert rejection_reasons(signals) == ["too_many_changed_lines", "too_many_files", "binary_diff"]
    assert rejection_reasons(signals, FilterCriteria(max_code_lines=2000, max_code_files=40)) == ["binary_diff"]

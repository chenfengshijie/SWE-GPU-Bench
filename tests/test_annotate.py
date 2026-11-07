import json

from swe_gpu_bench.collect.filter import ANNOTATION_VERSION, annotate_pr_file, annotate_prs


def write_pr(tmp_path, payload_extra=None):
    metadata_path = tmp_path / "prs" / "acme__project" / "7.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        json.dumps(
            {
                "repo": "acme/project",
                "number": 7,
                "title": "Fix parser",
                "body": "Fixes #7",
                "commit_messages": [],
                "merged_at": "2024-02-02T00:00:00Z",
                "base_ref": "main",
                "default_branch": "main",
                **(payload_extra or {}),
            }
        ),
        encoding="utf-8",
    )

    patch_path = tmp_path / "patches" / "acme__project" / "7.diff"
    patch_path.parent.mkdir(parents=True)
    patch_path.write_text(
        """diff --git a/csrc/parser.cu b/csrc/parser.cu
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
""",
        encoding="utf-8",
    )
    return metadata_path


def test_annotate_pr_file_adds_boolean_tags(tmp_path):
    metadata_path = write_pr(tmp_path)

    resolves_issues, has_test_patch = annotate_pr_file(metadata_path, tmp_path / "patches")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert resolves_issues is True
    assert has_test_patch is True
    assert payload["resolves_issues"] is True
    assert payload["has_test_patch"] is True
    assert payload["annotation_version"] == ANNOTATION_VERSION
    assert payload["merged"] is True
    assert payload["targets_default_branch"] is True
    assert payload["gpu_signal"] is True
    assert payload["code_files"] == 1
    assert payload["code_changed_lines"] == 2


def test_annotate_prs_refreshes_annotations_from_older_versions(tmp_path):
    metadata_path = write_pr(tmp_path, {"resolves_issues": False, "has_test_patch": False})

    first = annotate_prs(tmp_path / "prs", tmp_path / "patches")
    second = annotate_prs(tmp_path / "prs", tmp_path / "patches")

    assert first["skipped_already_annotated"] == 0
    assert first["has_test_patch"] == 1
    assert second["skipped_already_annotated"] == 1
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["gpu_signal"] is True

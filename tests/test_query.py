import argparse
import json
from pathlib import Path

from swe_gpu_bench.generate import query

DIFF = (
    "diff --git a/python/flashinfer/sampling.py b/python/flashinfer/sampling.py\n"
    "--- a/python/flashinfer/sampling.py\n"
    "+++ b/python/flashinfer/sampling.py\n"
    "@@ -1 +1,2 @@\n"
    "+def top_k_sampling_from_probs(probs, top_k, deterministic=True):\n"
    "+    return probs\n"
    "diff --git a/tests/test_sampling.py b/tests/test_sampling.py\n"
    "--- a/tests/test_sampling.py\n"
    "+++ b/tests/test_sampling.py\n"
    "@@ -1 +1,2 @@\n"
    "+def test_top_k():\n"
    "+    assert top_k_sampling_from_probs(x, top_k=3, deterministic=True) is not None\n"
)


def test_build_prompt_lists_interface_names():
    template = "Interface:\n{interface}\nPatch:\n{patch}"

    assert query.build_prompt(template, {}, "diff", 100, ["top_k", "sample"]) == "Interface:\n- top_k\n- sample\nPatch:\ndiff"
    assert query.build_prompt(template, {}, "diff", 100) == "Interface:\n(none)\nPatch:\ndiff"


def test_interface_names_come_from_the_split_patch():
    assert query.interface_names(DIFF, ["pytest tests/test_sampling.py"]) == [
        "deterministic",
        "top_k",
        "top_k_sampling_from_probs",
    ]


def test_generate_one_passes_interface_to_the_model(tmp_path: Path, monkeypatch):
    slug = "flashinfer-ai__flashinfer"
    (tmp_path / "prs" / slug).mkdir(parents=True)
    (tmp_path / "prs" / slug / "238.json").write_text(json.dumps({"repo": "flashinfer-ai/flashinfer", "title": "Add top-k"}))
    (tmp_path / "patches" / slug).mkdir(parents=True)
    (tmp_path / "patches" / slug / "238.diff").write_text(DIFF)
    prompts = []
    monkeypatch.setattr(query, "complete", lambda prompt, **kwargs: prompts.append(prompt) or "Add top-k sampling.")
    args = argparse.Namespace(
        prs_dir=str(tmp_path / "prs"),
        patches_dir=str(tmp_path / "patches"),
        max_patch_chars=50000,
        env_file=None,
        max_retries=1,
        retry_delay_seconds=0.0,
    )
    record = {"repo": "flashinfer-ai/flashinfer", "pr_number": 238, "test_commands": ["pytest tests/test_sampling.py"]}

    result = query.generate_one(record, args, "{title}\n{interface}", profile=None)

    assert result["error"] is None
    assert result["query"] == "Add top-k sampling."
    assert result["interface"] == ["deterministic", "top_k", "top_k_sampling_from_probs"]
    assert prompts == ["Add top-k\n- deterministic\n- top_k\n- top_k_sampling_from_probs"]

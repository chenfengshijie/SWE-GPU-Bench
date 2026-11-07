from swe_gpu_bench.core.diff import changed_lines, new_names_used_by_tests, summarize_diff


def section(path: str, *lines: str) -> str:
    body = "\n".join(lines)
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n{body}\n"


def test_changed_lines_skips_headers_and_context():
    text = section("src/a.py", " context", "-old", "+new")

    assert changed_lines(text) == ["old", "new"]
    assert changed_lines(text, ("+",)) == ["new"]


def test_summarize_diff_separates_tests_docs_and_code():
    text = (
        section("csrc/attn.cu", "-int x = 1;", "+int x = 2;")
        + section("docs/attn.md", "+Explains the kernel.")
        + section("CHANGELOG.md", "+- fix")
        + section("tests/test_attn.py", "+def test_attn():", "+    assert True")
    )

    summary = summarize_diff(text)

    assert summary.test_files == ("tests/test_attn.py",)
    assert summary.code_files == ("csrc/attn.cu",)
    assert summary.code_changed_lines == 2
    assert summary.gpu_signal is True
    assert summary.binary_files == ()


def test_summarize_diff_gpu_signal_from_paths_and_changed_lines():
    assert summarize_diff(section("lib/Dialect/TritonGPU/Transforms/Utility.cpp", "+int x;")).gpu_signal is True
    assert summarize_diff(section("runtime/nvqir/CuDensityMatState.cpp", "+int x;")).gpu_signal is True
    assert summarize_diff(section("third_party/amd/lib/Lowering.cpp", "+int x;")).gpu_signal is True
    assert summarize_diff(section("src/tokenizer.py", "+x = 1")).gpu_signal is False
    assert summarize_diff(section("src/tokenizer.py", '+x = torch.zeros(3, device="cuda")')).gpu_signal is True
    assert summarize_diff(section("docs/cuda.md", "+CUDA notes")).gpu_signal is False


def test_summarize_diff_reports_binary_files():
    text = "diff --git a/docs/plot.png b/docs/plot.png\nBinary files a/docs/plot.png and b/docs/plot.png differ\n"

    assert summarize_diff(text).binary_files == ("docs/plot.png",)


def test_new_names_used_by_tests_returns_names_the_patch_introduces():
    patch = section(
        "python/sampling.py",
        "-def sample(logits, top_k):",
        "+def top_k_sampling_from_probs(logits, top_k, deterministic=True):",
    )
    test_patch = section(
        "tests/test_sampling.py",
        "+def test_top_k():",
        "+    out = top_k_sampling_from_probs(x, top_k=3, deterministic=True)",
    )

    assert new_names_used_by_tests(patch, test_patch) == ["deterministic", "top_k_sampling_from_probs"]

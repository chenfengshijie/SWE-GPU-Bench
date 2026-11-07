from __future__ import annotations

from dataclasses import dataclass
import re


DIFF_HEADER_PAT = re.compile(r"^diff --git a/(.+?) b/(.+)$")
TEST_TOKENS = frozenset({"test", "tests", "testing", "testdata", "conftest", "unittest", "unittests", "e2e"})
PATH_TOKEN_SPLIT_PAT = re.compile(r"[._-]+")
CAMEL_CASE_TEST_PAT = re.compile(r"[a-z0-9]Tests?$")
BINARY_DIFF_PAT = re.compile(r"^Binary files .* differ$", re.MULTILINE)
DOC_PATH_PAT = re.compile(
    r"(?:^|/)docs?/|\.(?:md|rst|adoc)$|(?:^|/)(?:CHANGELOG|CHANGES|NEWS|AUTHORS|CONTRIBUTORS)[^/]*$", re.IGNORECASE
)

GPU_PATH_PAT = re.compile(
    r"\.(?:cu|cuh|ptx|hip|metal)$"
    r"|gpu|cuda|nvidia|nvptx|nvvm|rocdl|cudnn|cublas|cusparse|cufft|curand|cusolver|custatevec|cudensitymat|cutensor"
    r"|triton|cutlass|nccl|nvshmem"
    r"|(?:^|[/_.-])(?:amd|rocm|hip|rccl|cute|cub|libcudacxx|kernels?|csrc|sm_?\d{2,3}a?|hopper|blackwell|ampere"
    r"|gfx\d+|attention|attn|fmha|mla|gemm|moe|tma|wgmma|mma)(?:$|[/_.-])",
    re.IGNORECASE,
)
GPU_CODE_PAT = re.compile(
    r"(?i:gpu|cuda|nvptx|nvvm|rocdl|rocm|cudnn|cublas|nccl|triton)|\bhip[A-Z]\w*"
    r"|__global__|__device__|__shared__|__syncthreads|__shfl|<<<|\bthreadIdx\b|\bblockIdx\b|\bblockDim\b|\bgridDim\b"
    r"|\bwarp\w*|\btl\.\w+|\bcp\.async|\bwgmma\b|\btcgen05\b|\bmma\b|\bldmatrix\b|\bmbarrier\b"
)

PY_DEF_PAT = re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w{3,})\s*\(|^\s*class\s+([A-Za-z_]\w{3,})", re.MULTILINE)
PY_DEF_PARAMS_PAT = re.compile(r"^\s*(?:async\s+)?def\s+\w+\s*\(([^)]*)", re.MULTILINE)
BINDING_DEF_PAT = re.compile(r"""(?:\.def|TORCH_LIBRARY\w*|register\w*)\s*\(\s*["']([A-Za-z_]\w{3,})""")
IDENTIFIER_PAT = re.compile(r"[A-Za-z_]\w+")
COMMON_NAMES = frozenset({
    "self", "None", "True", "False", "torch", "tensor", "device", "dtype", "cuda", "return", "import", "from",
    "test", "tests", "assert", "kwargs", "args", "shape", "size", "output", "input", "value", "values", "float",
    "int", "bool", "list", "dict", "range", "print", "main", "init", "setup", "forward", "backward", "name",
    "data", "config", "result", "results", "batch", "seqlen", "head", "heads", "causal", "scale", "block",
})


@dataclass(frozen=True)
class DiffSection:
    old_path: str
    new_path: str
    raw_text: str


@dataclass(frozen=True)
class DiffSummary:
    test_files: tuple[str, ...]
    code_files: tuple[str, ...]
    code_changed_lines: int
    gpu_signal: bool
    binary_files: tuple[str, ...]


def is_test_like_path(path: str) -> bool:
    """A path segment (file name without extension) holds a whole test token: ``tests/``, ``test_x.py``, ``x_test.cc``, ``XTest.cpp``.

    Substrings do not count, so ``warpspecialized.hpp`` or ``latest.py`` are not tests.
    """
    parts = [part for part in path.replace("\\", "/").split("/") if part]
    for index, part in enumerate(parts):
        is_file = index == len(parts) - 1
        name = part.rsplit(".", 1)[0] if is_file and "." in part.lstrip(".") else part
        if TEST_TOKENS.intersection(PATH_TOKEN_SPLIT_PAT.split(name.lower())) or CAMEL_CASE_TEST_PAT.search(name):
            return True
    return False


def parse_diff_sections(text: str) -> list[DiffSection]:
    sections: list[DiffSection] = []
    current_lines: list[str] = []
    current_old_path: str | None = None
    current_new_path: str | None = None

    for line in text.splitlines(keepends=True):
        match = DIFF_HEADER_PAT.match(line.rstrip("\n"))
        if match:
            if current_lines and current_old_path is not None and current_new_path is not None:
                sections.append(
                    DiffSection(
                        old_path=current_old_path,
                        new_path=current_new_path,
                        raw_text="".join(current_lines),
                    )
                )
            current_lines = [line]
            current_old_path = match.group(1)
            current_new_path = match.group(2)
            continue
        if current_lines:
            current_lines.append(line)

    if current_lines and current_old_path is not None and current_new_path is not None:
        sections.append(
            DiffSection(
                old_path=current_old_path,
                new_path=current_new_path,
                raw_text="".join(current_lines),
            )
        )
    return sections


def diff_files(text: str) -> list[str]:
    return [section.new_path for section in parse_diff_sections(text)]


def changed_lines(text: str, prefixes: tuple[str, ...] = ("+", "-")) -> list[str]:
    """Added/removed hunk lines (without the marker); file headers and context lines are skipped."""
    lines: list[str] = []
    in_hunk = False
    for line in text.splitlines():
        if line.startswith("diff --git "):
            in_hunk = False
        elif line.startswith("@@"):
            in_hunk = True
        elif in_hunk and line[:1] in prefixes:
            lines.append(line[1:])
    return lines


def summarize_diff(text: str) -> DiffSummary:
    """Split a PR diff into test and code files; ``gpu_signal`` means a changed path or line looks GPU-specific."""
    test_files: list[str] = []
    code_files: list[str] = []
    binary_files: list[str] = []
    code_changed = 0
    gpu_signal = False
    for section in parse_diff_sections(text):
        path = section.new_path
        if BINARY_DIFF_PAT.search(section.raw_text):
            binary_files.append(path)
        is_test = is_test_like_path(path)
        if not is_test and DOC_PATH_PAT.search(path):
            continue
        lines = changed_lines(section.raw_text)
        if not gpu_signal:
            gpu_signal = bool(GPU_PATH_PAT.search(path)) or any(GPU_CODE_PAT.search(line) for line in lines)
        if is_test:
            test_files.append(path)
            continue
        code_files.append(path)
        code_changed += len(lines)
    return DiffSummary(
        test_files=tuple(test_files),
        code_files=tuple(code_files),
        code_changed_lines=code_changed,
        gpu_signal=gpu_signal,
        binary_files=tuple(binary_files),
    )


def new_names_used_by_tests(patch_text: str, test_patch_text: str) -> list[str]:
    """Functions, classes, parameters and bindings the patch introduces that the test patch's added lines use."""
    added = "\n".join(changed_lines(patch_text, ("+",)))
    removed = "\n".join(changed_lines(patch_text, ("-",)))
    names: set[str] = set()
    for match in PY_DEF_PAT.finditer(added):
        names.add(match.group(1) or match.group(2))
    names.update(BINDING_DEF_PAT.findall(added))
    for match in PY_DEF_PARAMS_PAT.finditer(added):
        for param in match.group(1).split(","):
            param = param.strip().split(":")[0].split("=")[0].strip().lstrip("*")
            if len(param) >= 4:
                names.add(param)
    existing = set(IDENTIFIER_PAT.findall(removed))
    used = set(IDENTIFIER_PAT.findall("\n".join(changed_lines(test_patch_text, ("+",)))))
    return sorted(
        name
        for name in names
        if name in used and name not in existing and name not in COMMON_NAMES and not name.startswith("_")
    )

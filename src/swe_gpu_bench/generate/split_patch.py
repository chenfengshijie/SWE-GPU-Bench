"""``generate split-patch``: split each PR diff into an implementation patch and a test patch.

A diff section goes to the test patch when it touches a test-like path, a file named by the
recorded ``test_commands``, or the source of a program those commands execute (an example binary,
a Python script, or an example program a ctest target is named after); or when it is test support
(helpers, or build files whose changed lines register a test target) next to such a file.
"""
from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from swe_gpu_bench.core.diff import DiffSection, changed_lines, is_test_like_path, parse_diff_sections
from swe_gpu_bench.core.io import load_text, load_verified_records, patch_path, repo_dir, write_text


logger = logging.getLogger(__name__)

PYTEST_FILE_PAT = re.compile(r"(?P<path>(?:repo/)?[^'\"\s]+(?:test|tests)[^'\"\s]*\.(?:py|pyi|cu|cuh|cc|cpp|cxx|c|h|hpp))(?:::[^'\"\s]+)?")
SCRIPT_FILE_PAT = re.compile(r"(?P<path>(?:repo/)?[^'\"\s]+(?:test|tests)[^'\"\s]*\.(?:py|pyi|cu|cuh|cc|cpp|cxx|c|h|hpp))")
IMPORT_FROM_PAT = re.compile(r"from\s+(?P<module>(?:tests?\.[A-Za-z0-9_\.]+|test_[A-Za-z0-9_\.]+))\s+import\b")
IMPORT_PAT = re.compile(r"import\s+(?P<module>(?:tests?\.[A-Za-z0-9_\.]+|test_[A-Za-z0-9_\.]+))\b")
BINARY_PAT = re.compile(r"(?P<path>\./(?:tests?/[^'\"\s]+|test_[^'\"\s]+))")
CTEST_PAT = re.compile(r"\bctest\b.*?(?:-R|--tests-regex)\s+['\"]?(?P<name>[^'\"\n]+)")
TEST_TARGET_PAT = re.compile(r"\b(?:add_test|gtest|pytest|ctest|test_|tests?/|add_executable\s*\(\s*test)", re.IGNORECASE)
EXECUTED_PROGRAM_PAT = re.compile(
    r"(?:^|&&|\|\|?|;|\(|\bthen\b|\bdo\b)\s*(?:[A-Za-z_]\w*=\S*\s+)*(?:(?:timeout\s+\S+|time|env|nice)\s+)*"
    r"(?P<path>(?:\.{1,2}/)+[^\s'\";|&()]+|[\w.-]+(?:/[^\s'\";|&()/]+)+)",
    re.MULTILINE,
)
PYTHON_SCRIPT_PAT = re.compile(r"\bpython[\d.]*\s+(?:-[A-Za-z]+\s+)*(?P<path>[^\s'\";|&()-][^\s'\";|&()]*\.py)\b")
LAUNCHER_NAMES = frozenset(
    {"python", "python3", "pip", "uv", "bash", "sh", "env", "activate", "cmake", "ctest", "make", "ninja",
     "pytest", "torchrun", "mpirun", "nsys", "ncu", "compute-sanitizer"}
)
PROGRAM_SOURCE_SUFFIXES = (".cu", ".cpp", ".cc", ".cxx", ".c")
HARNESS_DIR_NAMES = frozenset(
    {"example", "examples", "sample", "samples", "tutorial", "tutorials", "demo", "demos", "benchmark", "benchmarks"}
)
NEW_FILE_PAT = re.compile(r"^new file mode ", re.MULTILINE)

TEST_SUPPORT_FILE_NAMES = {
    "cmakelists.txt",
    "makefile",
    "build",
    "build.bazel",
}
TEST_SUPPORT_SUFFIXES = (".cmake",)
TEST_HELPER_NAME_PATS = (
    re.compile(r"(^|[/\\])cpu_reference\.[^/\\]+$", re.IGNORECASE),
    re.compile(r"(^|[/\\]).*test_utils?\.[^/\\]+$", re.IGNORECASE),
    re.compile(r"(^|[/\\]).*_test_utils?\.[^/\\]+$", re.IGNORECASE),
)


@dataclass(frozen=True)
class TestCommandSignals:
    test_files: frozenset[str]
    test_targets: frozenset[str]
    executables: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class SplitResult:
    patch_text: str
    test_patch_text: str


def normalize_repo_relative_path(path: str) -> str:
    normalized = path.strip().strip("\"'").replace("\\", "/")
    if normalized.startswith("repo/"):
        normalized = normalized[len("repo/") :]
    return normalized.lstrip("./")


def module_to_candidate_paths(module: str) -> list[str]:
    module = module.strip().strip(",")
    if not module:
        return []
    base = module.replace(".", "/")
    candidates = [f"{base}.py"]
    if not module.startswith("tests."):
        candidates.append(f"tests/{base}.py")
    return [normalize_repo_relative_path(candidate) for candidate in candidates]


def extract_test_signals(commands: list[str]) -> TestCommandSignals:
    test_files: set[str] = set()
    test_targets: set[str] = set()
    executables: set[str] = set()

    for command in commands:
        for match in EXECUTED_PROGRAM_PAT.finditer(command):
            name = Path(match.group("path")).name
            if name and "." not in name and name not in LAUNCHER_NAMES:
                executables.add(name)
        for match in PYTHON_SCRIPT_PAT.finditer(command):
            path = normalize_repo_relative_path(match.group("path"))
            if Path(path).name != "setup.py":
                test_files.add(path)
        for match in PYTEST_FILE_PAT.finditer(command):
            test_files.add(normalize_repo_relative_path(match.group("path")))
        for match in SCRIPT_FILE_PAT.finditer(command):
            path = normalize_repo_relative_path(match.group("path"))
            if is_test_like_path(path):
                test_files.add(path)
        for match in IMPORT_FROM_PAT.finditer(command):
            test_files.update(module_to_candidate_paths(match.group("module")))
        for match in IMPORT_PAT.finditer(command):
            test_files.update(module_to_candidate_paths(match.group("module")))
        for match in BINARY_PAT.finditer(command):
            raw_path = normalize_repo_relative_path(match.group("path"))
            test_targets.add(Path(raw_path).name)
            if is_test_like_path(raw_path):
                test_files.add(raw_path)
        for match in CTEST_PAT.finditer(command):
            raw_name = match.group("name").strip()
            test_targets.add(raw_name)
            for token in re.split(r"[^A-Za-z0-9_]+", raw_name):
                if token.startswith("test") or token.startswith("ctest"):
                    test_targets.add(token)

    return TestCommandSignals(
        test_files=frozenset(sorted(test_files)),
        test_targets=frozenset(sorted(target for target in test_targets | executables if target)),
        executables=frozenset(sorted(executables)),
    )


def collect_verified_targets(verified_dir: Path) -> dict[tuple[str, int], list[str]]:
    """``test_commands`` per verified (repo, pr)."""
    return {
        (record["repo"], int(record["pr_number"])): list(record.get("test_commands") or [])
        for record in load_verified_records(verified_dir)
    }


def is_support_file(path: str) -> bool:
    name = Path(path).name.lower()
    return name in TEST_SUPPORT_FILE_NAMES or path.lower().endswith(TEST_SUPPORT_SUFFIXES)


def _looks_like_test_helper(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return any(pattern.search(normalized) for pattern in TEST_HELPER_NAME_PATS)


def _content_mentions_targets(section: DiffSection, signals: TestCommandSignals) -> bool:
    changed = "\n".join(changed_lines(section.raw_text))
    if TEST_TARGET_PAT.search(changed):
        return True
    file_tokens = {Path(path).name for path in signals.test_files}
    tokens = file_tokens | set(signals.test_targets)
    lowered = changed.lower()
    return any(token and token.lower() in lowered for token in tokens)


def _same_directory_as_test_file(section: DiffSection, signals: TestCommandSignals) -> bool:
    section_dirs = {str(Path(section.old_path).parent), str(Path(section.new_path).parent)}
    test_dirs = {str(Path(path).parent) for path in signals.test_files}
    return bool(section_dirs & test_dirs)


def _named_by_commands(path: str, signals: TestCommandSignals) -> bool:
    """A file the commands name, the source of a binary they run, or an example a ctest target is named after."""
    if any(path == name or path.endswith("/" + name) for name in signals.test_files):
        return True
    candidate = Path(path)
    if candidate.suffix not in PROGRAM_SOURCE_SUFFIXES:
        return False
    if candidate.stem in signals.executables:
        return True
    if not HARNESS_DIR_NAMES.intersection(part.lower() for part in candidate.parent.parts):
        return False
    name_pat = re.compile(rf"(?<![A-Za-z0-9]){re.escape(candidate.stem)}(?![A-Za-z0-9])")
    return any(name_pat.search(target) for target in signals.test_targets)


def is_direct_test_section(section: DiffSection, signals: TestCommandSignals) -> bool:
    paths = {normalize_repo_relative_path(section.old_path), normalize_repo_relative_path(section.new_path)}
    return any(is_test_like_path(path) or _named_by_commands(path, signals) for path in paths)


def is_support_test_section(section: DiffSection, signals: TestCommandSignals, has_direct_test: bool) -> bool:
    if not has_direct_test:
        return False
    paths = [normalize_repo_relative_path(section.old_path), normalize_repo_relative_path(section.new_path)]
    if any(_looks_like_test_helper(path) for path in paths) and _same_directory_as_test_file(section, signals):
        return True
    if any(is_support_file(path) for path in paths) and _content_mentions_targets(section, signals):
        return True
    return False


def _harness_dirs(sections: list[DiffSection]) -> set[str]:
    """Directories of example/benchmark programs the test commands run (not already test-like paths)."""
    dirs: set[str] = set()
    for section in sections:
        path = normalize_repo_relative_path(section.new_path)
        parent = Path(path).parent
        if not is_test_like_path(path) and HARNESS_DIR_NAMES.intersection(part.lower() for part in parent.parts):
            dirs.add(parent.as_posix())
    return dirs


def _is_new_file_in(section: DiffSection, dirs: set[str]) -> bool:
    if not dirs or not NEW_FILE_PAT.search(section.raw_text):
        return False
    path = normalize_repo_relative_path(section.new_path)
    return any(path.startswith(directory + "/") for directory in dirs)


def split_patch_text(diff_text: str, signals: TestCommandSignals) -> SplitResult:
    sections = parse_diff_sections(diff_text)
    direct_test_indices = {
        index for index, section in enumerate(sections) if is_direct_test_section(section, signals)
    }
    harness_dirs = _harness_dirs([sections[index] for index in direct_test_indices])
    test_patch_parts: list[str] = []
    patch_parts: list[str] = []

    for index, section in enumerate(sections):
        if (
            index in direct_test_indices
            or _is_new_file_in(section, harness_dirs)
            or is_support_test_section(section, signals, has_direct_test=bool(direct_test_indices))
        ):
            test_patch_parts.append(section.raw_text)
        else:
            patch_parts.append(section.raw_text)

    return SplitResult(
        patch_text="".join(patch_parts),
        test_patch_text="".join(test_patch_parts),
    )


def split_verified_patches(verified_dir: Path, patches_dir: Path, output_dir: Path) -> tuple[int, int]:
    processed = 0
    missing = 0
    for (repo, pr_number), commands in collect_verified_targets(verified_dir).items():
        patch_file = patch_path(patches_dir, repo, pr_number)
        if not patch_file.exists():
            logger.warning("Missing patch for %s#%s at %s", repo, pr_number, patch_file)
            missing += 1
            continue
        split = split_patch_text(load_text(patch_file), extract_test_signals(commands))
        out_dir = repo_dir(output_dir, repo)
        write_text(out_dir / f"{pr_number}.diff", split.patch_text)
        write_text(out_dir / f"{pr_number}.test.diff", split.test_patch_text)
        processed += 1
    return processed, missing


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--verified-dir", default="data/verified")
    parser.add_argument("--patches-dir", default="data/patches")
    parser.add_argument("--output-dir", default="data/split_patch")


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    processed, missing = split_verified_patches(
        Path(args.verified_dir),
        Path(args.patches_dir),
        Path(args.output_dir),
    )
    logger.info("Wrote split patches for %s PRs (%s missing patches)", processed, missing)
    return 0

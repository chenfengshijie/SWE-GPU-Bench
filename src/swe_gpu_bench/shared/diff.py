from __future__ import annotations

from dataclasses import dataclass
import re


DIFF_HEADER_PAT = re.compile(r"^diff --git a/(.+?) b/(.+)$")
TEST_PATH_HINTS = ("test", "tests", "testing", "e2e", "spec")


@dataclass(frozen=True)
class DiffSection:
    old_path: str
    new_path: str
    raw_text: str


def is_test_like_path(path: str) -> bool:
    lowered = path.lower()
    parts = re.split(r"[/\\._-]+", lowered)
    # Match path/name tokens, never substrings: "specializations" is production
    # code, not a "spec" test; likewise "latest" and "contest" are not tests.
    return any(hint in parts for hint in TEST_PATH_HINTS)


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

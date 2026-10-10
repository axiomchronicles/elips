#!/usr/bin/env python3
"""Extract and execute ``python`` code blocks from the ELIPS docs tree.

This is the Phase-8 *harness*; the docs it runs over are Phase 9. It proves
that documented Python snippets actually run against the built ``elips``
module, failing CI on any error.

Rules
-----
* Only fenced ``python`` / ``py`` blocks are considered (``bash`` etc. ignored).
* A block is **skipped** when it is marked no-exec, either on the fence info
  string (```` ```python no-exec ````) or on the block's first comment line
  (``# no-exec`` / ``# doctest: +SKIP``). Phase 9 docs mark pseudocode or
  network/filesystem snippets this way.
* Blocks in one file share a cumulative namespace (like doctest), so a later
  block may use names a prior block defined.

Usage::

    python scripts/run_doc_snippets.py                 # scan docs/
    python scripts/run_doc_snippets.py docs/_redesign/_snippets_seed.md
    python scripts/run_doc_snippets.py --list docs/python
    python scripts/run_doc_snippets.py --selftest      # verify the parser
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import traceback
from dataclasses import dataclass

# ```lang [info]\n ... \n``` — capture language, info string and body with its
# starting line number so failures point back at the doc.
_FENCE = re.compile(
    r"^(?P<indent>[ \t]*)```(?P<lang>[A-Za-z0-9_+-]*)[ \t]*(?P<info>[^\n]*)\n"
    r"(?P<body>.*?)^(?P=indent)```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
_SKIP = re.compile(r"no[-_ ]?exec|noexec|doctest:\s*\+skip", re.IGNORECASE)


@dataclass
class Block:
    path: str
    line: int  # 1-based line of the opening fence
    code: str
    skip: bool


def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def extract(path: str) -> list[Block]:
    """Return every python code block found in a markdown file."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    blocks: list[Block] = []
    for m in _FENCE.finditer(text):
        if m.group("lang").lower() not in ("python", "py"):
            continue
        body = m.group("body")
        first = next((ln for ln in body.splitlines() if ln.strip()), "")
        skip = bool(_SKIP.search(m.group("info"))) or (
            first.lstrip().startswith("#") and bool(_SKIP.search(first))
        )
        blocks.append(Block(path, _line_of(text, m.start()), body, skip))
    return blocks


def find_markdown(roots: list[str]) -> list[str]:
    files: list[str] = []
    for root in roots:
        if os.path.isfile(root):
            files.append(root)
        else:
            files.extend(
                sorted(glob.glob(os.path.join(root, "**", "*.md"), recursive=True))
            )
    return files


def run(roots: list[str], *, list_only: bool = False, allow_fail: bool = False) -> int:
    files = find_markdown(roots)
    executed = skipped = failed = 0
    for path in files:
        ns: dict = {"__name__": "__doc_snippet__"}
        for blk in extract(path):
            if blk.skip:
                skipped += 1
                continue
            if list_only:
                print(f"  exec {path}:{blk.line}")
                executed += 1
                continue
            try:
                exec(compile(blk.code, f"{path}:{blk.line}", "exec"), ns)
                executed += 1
            except Exception:  # noqa: BLE001 — report every doc failure
                failed += 1
                print(f"\n[FAIL] {path}:{blk.line}", file=sys.stderr)
                traceback.print_exc()

    print(
        f"\nsnippets: {executed} executed, {skipped} skipped (no-exec), "
        f"{failed} failed  across {len(files)} file(s)"
    )
    if executed == 0 and skipped == 0:
        print("no python snippets found — nothing to run (ok)")
    return 1 if (failed and not allow_fail) else 0


def _selftest() -> int:
    import tempfile

    md = (
        "# t\n"
        "```python\nx = 21 * 2\nassert x == 42\n```\n"
        "```python no-exec\nthis is pseudocode, not real()\n```\n"
        "```bash\necho ignored\n```\n"
    )
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.md")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(md)
        blocks = extract(p)
        assert len(blocks) == 2, blocks  # bash ignored
        assert blocks[0].skip is False
        assert blocks[1].skip is True  # no-exec on fence info
        assert run([p]) == 0  # the one real block passes
    # first-line marker is honored too
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "u.md")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("```python\n# no-exec\nboom(\n```\n")
        assert extract(p)[0].skip is True
    print("selftest OK")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "roots",
        nargs="*",
        default=["docs"],
        help="markdown files or directories (default: docs)",
    )
    ap.add_argument("--list", action="store_true", help="list blocks, don't run")
    ap.add_argument(
        "--allow-fail", action="store_true", help="always exit 0 (report-only)"
    )
    ap.add_argument("--selftest", action="store_true", help="verify the parser")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    return run(args.roots or ["docs"], list_only=args.list, allow_fail=args.allow_fail)


if __name__ == "__main__":
    sys.exit(main())

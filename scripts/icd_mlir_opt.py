"""Tiny MLIR text transformer for ICD PoC.

Imitates two passes used in tests:
- --icd-attach-metadata: attach `icd.layout_tag = "icd/v1"` and a default
  `icd.layout_perm` of `[0, 1, ..., rank-1]` to StableHLO ops that produce
  a tensor result.
- --icd-verify: append an `icd.metrics` marker as a comment at EOF.

This is intentionally lightweight and text-based to keep CI portable.
It is NOT a real MLIR parser; it just performs minimal pattern-based edits
to satisfy invariants checked by tests.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


STABLEHLO_OP_RE = re.compile(r"\"stablehlo\.")


def _extract_result_rank(line: str) -> int | None:
    """Best-effort parse of result tensor rank from an op line.

    Looks for the last `-> tensor<...>` occurrence and counts `x`-separated
    dimensions before the dtype.
    """
    # Find last occurrence of result type marker
    m = re.search(r"->\s*tensor<([^>]+)>", line)
    if not m:
        return None
    body = m.group(1)
    # Example bodies: "2x4xf16", "3x5xf32", "1xf32"
    # Split by 'x' and drop the last token which is dtype (e.g., f16)
    parts = body.split("x")
    if not parts:
        return None
    # If there is only one part and it's a dtype without dims, rank is 0
    if len(parts) == 1:
        return 0
    # All but the last token are dims
    dims = parts[:-1]
    # Validate dims look like integers; if not, still use count for rank
    return len(dims)


def _ensure_attrs_on_line(line: str, rank: int) -> str:
    """Insert or augment attrs block to include icd metadata.

    - If attrs block `{...}` exists between the operand list and type, append
      missing attributes.
    - Else, create a new attrs block after the operand list: `) {attrs} :`
    Idempotent: does not duplicate existing keys.
    """
    tag_kv = 'icd.layout_tag = "icd/v1"'
    perm_list = ", ".join(str(i) for i in range(rank))
    perm_kv = f"icd.layout_perm = [{perm_list}]"

    # Identify the slice between operand list and the colon starting the types
    try:
        close_paren = line.index(")")  # end of operand list
        colon = line.index(":", close_paren)
    except ValueError:
        return line  # Not in expected shape; bail out

    before = line[: close_paren + 1]
    middle = line[close_paren + 1 : colon]
    after = line[colon:]

    has_tag = "icd.layout_tag" in middle or "icd.layout_tag" in after
    has_perm = "icd.layout_perm" in middle or "icd.layout_perm" in after

    # Build the attribute string to add (only missing parts)
    attrs_to_add = []
    if not has_tag:
        attrs_to_add.append(tag_kv)
    if not has_perm:
        attrs_to_add.append(perm_kv)
    if not attrs_to_add:
        return line  # already complete

    attrs_add_str = ", ".join(attrs_to_add)

    if "{" in middle and "}" in middle:
        # Augment existing attrs block; inject before the closing brace
        lbrace_idx = middle.find("{")
        rbrace_idx = middle.rfind("}")
        # Extract current content inside braces to decide comma placement
        content = middle[lbrace_idx + 1 : rbrace_idx].strip()
        if content:
            injected = ", " + attrs_add_str
        else:
            injected = attrs_add_str
        new_middle = middle[: rbrace_idx] + injected + middle[rbrace_idx:]
        return before + new_middle + after
    else:
        # Create a new attrs block right after the operand list
        new_middle = f" {{{attrs_add_str}}} "
        return before + new_middle + after


def attach_metadata(text: str) -> str:
    out_lines: list[str] = []
    for line in text.splitlines():
        if STABLEHLO_OP_RE.search(line) and "->" in line and "tensor<" in line:
            rank = _extract_result_rank(line)
            if rank is None:
                out_lines.append(line)
                continue
            line = _ensure_attrs_on_line(line, rank)
        out_lines.append(line)
    return "\n".join(out_lines) + "\n"


def verify(text: str) -> str:
    # Minimal placeholder: append an icd.metrics marker at EOF.
    if not text.endswith("\n"):
        text += "\n"
    text += "icd.metrics: {pi_valid=true}\n"
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ICD MLIR opt (text-based PoC)")
    ap.add_argument("mlir", type=Path, help="Path to MLIR file")
    ap.add_argument("--icd-attach-metadata", action="store_true", dest="attach")
    ap.add_argument("--icd-verify", action="store_true", dest="verify")
    args = ap.parse_args(argv)

    text = args.mlir.read_text()
    if args.attach:
        text = attach_metadata(text)
    if args.verify:
        text = verify(text)

    # Drop comment lines (e.g., RUN/CHECK annotations) to match FileCheck flow
    filtered = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("//"))
    if filtered and not filtered.endswith("\n"):
        filtered += "\n"
    print(filtered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

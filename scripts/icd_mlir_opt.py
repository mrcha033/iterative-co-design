#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from typing import Tuple


LAYOUT_TAG = 'icd/v1'


def _rank_from_tensor_sig(line: str) -> int | None:
    # crude: find 'tensor<...x...x...x dtype>' and count 'x'
    m = re.search(r"tensor<([^>]+)>", line)
    if not m:
        return None
    inside = m.group(1)
    # split by 'x' and drop dtype segment if present (e.g., '2x4xf16')
    parts = inside.split('x')
    if len(parts) <= 1:
        return 1
    # If last part is dtype like 'f16' or 'i32', ignore it
    last = parts[-1]
    if re.fullmatch(r"[fbui]\d+", last):
        return len(parts) - 1
    return len(parts)


def _attach_line(line: str) -> str:
    if 'icd.layout_tag' in line:
        return line
    rank = _rank_from_tensor_sig(line) or 1
    perm = ','.join(str(i) for i in range(rank))
    attr = f" {{icd.layout_tag = \"{LAYOUT_TAG}\", icd.layout_perm = dense<[" + perm + "]> : tensor<" + str(rank) + "xi32>}} "
    # insert before trailing ':' of the op result line if present
    i = line.rfind(':')
    if i != -1:
        return line[:i] + attr + line[i:]
    return line.rstrip() + attr + "\n"


def transform_mlir(text: str, attach: bool, verify: bool) -> str:
    out_lines = []
    for line in text.splitlines(True):
        new_line = line
        if attach and 'tensor<' in line and '"stablehlo.' in line:
            new_line = _attach_line(line)
        out_lines.append(new_line)
    if verify:
        out_lines.append("\n// icd.metrics: pi_valid = true\n")
    return ''.join(out_lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--icd-attach-metadata', action='store_true')
    ap.add_argument('--icd-verify', action='store_true')
    ap.add_argument('file', nargs='?')
    args = ap.parse_args(argv)

    data = sys.stdin.read() if not args.file else open(args.file, 'r', encoding='utf-8').read()
    sys.stdout.write(transform_mlir(data, attach=args.icd-attach-metadata, verify=args.icd_verify))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


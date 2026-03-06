#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare Python/Rust generation dump JSON files")
    ap.add_argument("--python-dump", required=True)
    ap.add_argument("--rust-dump", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--window", type=int, default=16)
    args = ap.parse_args()

    from tokenizers import Tokenizer  # type: ignore

    py = json.loads(Path(args.python_dump).read_text())
    rs = json.loads(Path(args.rust_dump).read_text())
    tok = Tokenizer.from_file(args.tokenizer)

    py_ids = py["generated_token_ids"]
    rs_ids = rs["generated_token_ids"]

    common = 0
    for a, b in zip(py_ids, rs_ids):
        if a != b:
            break
        common += 1

    print(f"python_stop_reason={py.get('stop_reason')} python_len={len(py_ids)} python_last={py.get('last_token_id')}")
    print(f"rust_stop_reason={rs.get('stop_reason')} rust_len={len(rs_ids)} rust_last={rs.get('last_token_id')}")
    print(f"common_prefix={common}")

    def dump(label: str, seq: list[int]) -> None:
        print(f"\n{label}:")
        for token_id in seq:
            piece = tok.decode([token_id], skip_special_tokens=False)
            print(f"{token_id}\t{piece!r}")

    dump("python_next", py_ids[common : common + args.window])
    dump("rust_next", rs_ids[common : common + args.window])

    print("\npython_output_tail:")
    print(repr(py.get("output", "")[-240:]))
    print("\nrust_output_tail:")
    print(repr(rs.get("output", "")[-240:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

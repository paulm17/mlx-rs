#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


def load_python_trace(path: Path):
    out = {}
    with safe_open(path, framework="np") as f:
        for key in f.keys():
            out[key] = np.asarray(f.get_tensor(key), dtype=np.float32)
    return out


def load_rust_trace(manifest_path: Path):
    manifest = json.loads(manifest_path.read_text())
    base = manifest_path.parent
    out = {}
    for key, entry in manifest["tensors"].items():
        arr = np.fromfile(base / entry["file"], dtype="<f4")
        out[key] = arr.reshape(tuple(entry["shape"]))
    return out


def compare_tensor(name, py, rs, atol):
    if py.shape != rs.shape:
        return {
            "ok": False,
            "reason": f"shape mismatch python={py.shape} rust={rs.shape}",
        }
    diff = np.abs(py - rs)
    finite = np.isfinite(diff)
    comparable = diff[finite]
    if comparable.size == 0:
        max_diff = 0.0 if np.array_equal(py, rs) else float("inf")
        mse = 0.0 if np.array_equal(py, rs) else float("inf")
    else:
        max_diff = float(np.max(comparable))
        mse = float(np.mean(comparable * comparable))
    bad = diff > atol
    bad_count = int(np.count_nonzero(bad))
    ok = bad_count == 0 and np.array_equal(np.isfinite(py), np.isfinite(rs))
    result = {
        "ok": ok,
        "shape": py.shape,
        "mse": mse,
        "max_diff": max_diff,
        "bad_count": bad_count,
        "total": int(diff.size),
    }
    if not ok:
        flat_bad = np.flatnonzero(bad.reshape(-1))
        if flat_bad.size:
            idx = int(flat_bad[0])
            result["first_bad_flat_index"] = idx
            result["python_value"] = float(py.reshape(-1)[idx])
            result["rust_value"] = float(rs.reshape(-1)[idx])
        else:
            result["reason"] = "non-finite pattern mismatch"
    return result


def main():
    ap = argparse.ArgumentParser(description="Compare DiffusionGemma Python/Rust trace tensors")
    ap.add_argument("--python-trace", required=True)
    ap.add_argument("--rust-manifest", required=True)
    ap.add_argument("--atol", type=float, default=1e-2)
    ap.add_argument("--all", action="store_true", help="continue after first mismatch")
    args = ap.parse_args()

    py = load_python_trace(Path(args.python_trace))
    rs = load_rust_trace(Path(args.rust_manifest))
    common = sorted(set(py) & set(rs))
    missing_py = sorted(set(rs) - set(py))
    missing_rs = sorted(set(py) - set(rs))

    if missing_py:
        print("Only in Rust:")
        for key in missing_py:
            print(f"  {key}")
    if missing_rs:
        print("Only in Python:")
        for key in missing_rs:
            print(f"  {key}")

    first_failure = None
    for key in common:
        result = compare_tensor(key, py[key], rs[key], args.atol)
        status = "OK" if result["ok"] else "FAIL"
        print(
            f"{status} {key}: shape={result.get('shape')} "
            f"mse={result.get('mse')} max={result.get('max_diff')} "
            f"bad={result.get('bad_count')}/{result.get('total')}"
        )
        if not result["ok"]:
            print(json.dumps(result, indent=2, default=str))
            first_failure = key
            if not args.all:
                break

    if first_failure:
        raise SystemExit(f"first divergent tensor: {first_failure}")
    if missing_py or missing_rs:
        raise SystemExit("trace key sets differ")
    print("All common tensors match")


if __name__ == "__main__":
    main()

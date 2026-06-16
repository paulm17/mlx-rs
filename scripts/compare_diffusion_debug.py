#!/usr/bin/env python3
"""Compare Python vs Rust DiffusionGemma traces layer by layer.

Loads traces from both implementations, computes per-tensor MSE/max-diff,
and pinpoints exactly where the divergence starts.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load_safetensors(path: str) -> dict[str, np.ndarray]:
    from safetensors.numpy import load_file
    return load_file(path)


def compare_tensors(a: np.ndarray, b: np.ndarray, name: str) -> dict:
    """Compare two tensors and return detailed stats."""
    if a.shape != b.shape:
        return {
            "name": name,
            "status": "SHAPE_MISMATCH",
            "a_shape": list(a.shape),
            "b_shape": list(b.shape),
        }

    a = a.astype(np.float64)
    b = b.astype(np.float64)
    diff = np.abs(a - b)
    return {
        "name": name,
        "status": "OK",
        "shape": list(a.shape),
        "elements": int(a.size),
        "mse": float(np.mean(diff ** 2)),
        "max_diff": float(np.max(diff)),
        "mean_diff": float(np.mean(diff)),
        "a_range": [float(np.min(a)), float(np.max(a))],
        "b_range": [float(np.min(b)), float(np.max(b))],
        "a_mean": float(np.mean(a)),
        "b_mean": float(np.mean(b)),
        "cosine_sim": float(
            np.dot(a.ravel(), b.ravel())
            / (np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()) + 1e-12)
        ),
    }


def main():
    ap = argparse.ArgumentParser(description="Compare Python vs Rust DiffusionGemma traces")
    ap.add_argument("--python-dir", default="/tmp/diffusion_gemma_debug")
    ap.add_argument("--rust-dir", default="/tmp/diffusion_gemma_debug/rust")
    ap.add_argument("--threshold", type=float, default=1e-3,
                    help="Max-diff threshold for PASS/FAIL")
    args = ap.parse_args()

    py_dir = Path(args.python_dir)
    rs_dir = Path(args.rust_dir)

    py_tensors = load_safetensors(str(py_dir / "python_trace.safetensors"))
    rs_manifest_path = rs_dir / "rust_trace_manifest.json"
    if not rs_manifest_path.exists():
        print(f"ERROR: Rust manifest not found at {rs_manifest_path}")
        return

    with open(rs_manifest_path) as f:
        rs_manifest = json.load(f)

    rs_tensors = {}
    for name, entry in rs_manifest["tensors"].items():
        f32_path = rs_dir / entry["file"]
        if f32_path.exists():
            raw = np.fromfile(str(f32_path), dtype=np.float32)
            rs_tensors[name] = raw.reshape(entry["shape"])

    print("=" * 80)
    print("DIFFUSIONGEMMA PARITY COMPARISON")
    print("=" * 80)

    # Group tensors by prefix
    groups = {}
    for name in sorted(set(list(py_tensors.keys()) + list(rs_tensors.keys()))):
        prefix = name.rsplit(".", 1)[0] if "." in name else name
        groups.setdefault(prefix, []).append(name)

    all_results = []
    layer_results = {}

    for name in sorted(set(list(py_tensors.keys()) + list(rs_tensors.keys()))):
        if name not in py_tensors:
            print(f"  SKIP {name}: only in Python")
            continue
        if name not in rs_tensors:
            print(f"  SKIP {name}: only in Rust")
            continue

        result = compare_tensors(py_tensors[name], rs_tensors[name], name)
        all_results.append(result)

        # Track per-layer results
        if "decoder.layer_" in name:
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p.startswith("layer_"):
                    layer_key = ".".join(parts[:i+1])
                    layer_results.setdefault(layer_key, []).append(result)
                    break

    # Print per-layer summary
    print("\n" + "=" * 80)
    print("PER-LAYER DECODER SUMMARY")
    print("=" * 80)
    print(f"{'Layer':<40} {'Max Diff':>12} {'MSE':>12} {'Cosine':>12} {'Status':>8}")
    print("-" * 80)

    critical_layer = None
    for layer_key in sorted(layer_results.keys()):
        results = layer_results[layer_key]
        worst = max(results, key=lambda r: r.get("max_diff", 0))
        max_diff = worst.get("max_diff", 0)
        mse = worst.get("mse", 0)
        cosine = worst.get("cosine_sim", 1.0)
        status = "PASS" if max_diff < args.threshold else "FAIL"
        if status == "FAIL" and critical_layer is None:
            critical_layer = layer_key
        print(f"  {layer_key:<38} {max_diff:>12.6f} {mse:>12.8f} {cosine:>12.8f} {status:>8}")

    # Print detailed results for critical tensors
    print("\n" + "=" * 80)
    print("DETAILED RESULTS (sorted by max_diff)")
    print("=" * 80)
    all_results.sort(key=lambda r: r.get("max_diff", 0), reverse=True)
    print(f"{'Tensor':<50} {'Max Diff':>12} {'MSE':>12} {'Cosine':>12} {'Status':>8}")
    print("-" * 80)
    for r in all_results[:40]:
        max_diff = r.get("max_diff", 0)
        mse = r.get("mse", 0)
        cosine = r.get("cosine_sim", 0)
        status = r["status"]
        name = r["name"]
        if len(name) > 48:
            name = "..." + name[-45:]
        print(f"  {name:<48} {max_diff:>12.6f} {mse:>12.8f} {cosine:>12.8f} {status:>8}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total = len(all_results)
    passed = sum(1 for r in all_results if r.get("max_diff", 0) < args.threshold)
    failed = total - passed
    print(f"  Total tensors: {total}")
    print(f"  PASS (max_diff < {args.threshold}): {passed}")
    print(f"  FAIL: {failed}")

    if critical_layer:
        print(f"\n  *** CRITICAL: First divergence at: {critical_layer} ***")
        print(f"  The decoder produces different outputs starting from this layer.")
        print(f"  Check RoPE offset, attention masks, MoE path, and layer norms.")

    # Export results as JSON
    out_path = Path(args.python_dir) / "comparison_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "threshold": args.threshold,
            "total": total,
            "passed": passed,
            "failed": failed,
            "critical_layer": critical_layer,
            "results": all_results,
        }, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()

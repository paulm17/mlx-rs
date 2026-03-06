#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


DEFAULT_SOURCE = Path(
    "/Volumes/Data/Users/paul/.cache/huggingface/hub/models--LiquidAI--LFM2-24B-A2B-MLX-4bit/snapshots/fb67c8c23d38cd4d7a9a6415ab80eefe83feecae"
)
DEFAULT_DEST = Path("tmp_models/LFM2-24B-A2B-MLX-4bit-python-port")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a local shim model dir for the alternate LFM2 SwitchGLU Rust port."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Original LFM2 snapshot directory",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="Workspace-local shim model directory to create",
    )
    return parser.parse_args()


def ensure_symlink(src: Path, dest: Path) -> None:
    if dest.exists() or dest.is_symlink():
        if dest.is_dir() and not dest.is_symlink():
            shutil.rmtree(dest)
        else:
            dest.unlink()
    os.symlink(src, dest)


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    dest = args.dest.resolve()

    if not source.exists():
        raise SystemExit(f"source model directory does not exist: {source}")

    dest.mkdir(parents=True, exist_ok=True)

    config_path = source / "config.json"
    if not config_path.exists():
        raise SystemExit(f"missing config.json in source model directory: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["model_type"] = "lfm2_moe_python_port"
    config["architectures"] = ["Lfm2MoePythonPortForCausalLM"]
    (dest / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    for child in source.iterdir():
        if child.name == "config.json":
            continue
        ensure_symlink(child, dest / child.name)

    print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

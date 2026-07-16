#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pure_grpo.utils.config import load_config, parse_override, set_by_dotted_key
from pure_grpo.training.grpo_trainer import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a pure GRPO baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], help="Override a dotted config key, e.g. training.batch_size=2")
    args = parser.parse_args()
    config = load_config(args.config)
    for override in args.set:
        key, value = parse_override(override)
        set_by_dotted_key(config, key, value)
    run_training(config)


if __name__ == "__main__":
    main()

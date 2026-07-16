#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flashgrpo_b200.training.flashgrpo_trainer import run_training
from flashgrpo_b200.utils.config import load_config, parse_override, set_by_dotted_key


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny FlashGRPO smoke test")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config)
    smoke_overrides = {
        "run_name": "flashgrpo_smoke",
        "training.batch_size": 1,
        "training.accumulation_steps": 1,
        "training.num_epochs": 1,
        "training.sample_num": 2,
        "training.num_workers": 0,
        "training.save_steps": 0,
        "generation.repeated_generate_nums": 2,
        "generation.max_length": 256,
        "flashgrpo.max_tree_nodes_per_seq": 4,
        "flashgrpo.cpeak_nodes": 8,
        "flashgrpo.require_pretrained_heads": False,
        "flashgrpo.allow_random_init": True,
        "flashgrpo.medusa_heads_checkpoint": "",
        "flashgrpo.medusa_microbatch_size": 1,
        "flashgrpo.medusa_max_tokens_per_update": 512,
        "logging.log_dir": "logs/flashgrpo/flashgrpo_smoke",
        "training.saved_model_dir": "outputs/flashgrpo_smoke/target",
        "training.saved_medusa_dir": "outputs/flashgrpo_smoke/medusa",
    }
    for key, value in smoke_overrides.items():
        set_by_dotted_key(config, key, value)
    for override in args.set:
        key, value = parse_override(override)
        set_by_dotted_key(config, key, value)
    run_training(config)


if __name__ == "__main__":
    main()

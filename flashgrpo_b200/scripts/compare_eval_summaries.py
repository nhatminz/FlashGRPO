#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(value, digits=4):
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a compact comparison table for eval summary JSON files.")
    parser.add_argument("summaries", nargs="+")
    args = parser.parse_args()
    rows = []
    for path in args.summaries:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rows.append(data)
    cols = [
        ("name", "name"),
        ("n", "num_examples"),
        ("acc", "answer_accuracy"),
        ("format", "format_rate"),
        ("reward", "mean_combined_reward"),
        ("avg_tok", "avg_new_tokens"),
        ("tok/s", "tokens_per_second"),
        ("gen_h", "generation_time_s"),
        ("wall_h", "wall_time_s"),
    ]
    table = []
    for row in rows:
        item = []
        for _, key in cols:
            value = row.get(key, "")
            if key.endswith("_time_s"):
                value = float(value) / 3600.0
            item.append(fmt(value))
        table.append(item)
    widths = [max(len(header), *(len(row[idx]) for row in table)) for idx, (header, _) in enumerate(cols)]
    print("  ".join(header.ljust(widths[idx]) for idx, (header, _) in enumerate(cols)))
    print("  ".join("-" * width for width in widths))
    for row in table:
        print("  ".join(row[idx].ljust(widths[idx]) for idx in range(len(cols))))


if __name__ == "__main__":
    main()

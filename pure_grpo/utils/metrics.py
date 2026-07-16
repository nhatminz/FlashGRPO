import csv
import json
import os
from pathlib import Path
from typing import Any


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), child, out)
    elif isinstance(value, (list, tuple)):
        out[prefix] = json.dumps(value)
    else:
        out[prefix] = value


def flatten_metrics(row: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    _flatten("", row, flat)
    return flat


class MetricsLogger:
    def __init__(self, log_dir: str | os.PathLike[str], *, append: bool = False):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / "metrics.jsonl"
        self.csv_path = self.log_dir / "metrics.csv"
        self.rows: list[dict[str, Any]] = []
        self.fieldnames: list[str] = []
        if append and self.jsonl_path.exists():
            with self.jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        flat = flatten_metrics(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                    self.rows.append(flat)
                    self.fieldnames = sorted(set(self.fieldnames).union(flat))
            if self.rows:
                self._rewrite_csv()
            else:
                self.csv_path.write_text("")
        else:
            self.jsonl_path.write_text("")
            self.csv_path.write_text("")

    def log(self, row: dict[str, Any]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
        flat = flatten_metrics(row)
        self.rows.append(flat)
        new_fields = sorted(set(flat) - set(self.fieldnames))
        if new_fields:
            self.fieldnames = sorted(set(self.fieldnames).union(new_fields))
            self._rewrite_csv()
        else:
            with self.csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
                writer.writerow(flat)

    def _rewrite_csv(self) -> None:
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)

    def write_summary(self, text: str) -> None:
        (self.log_dir / "summary.txt").write_text(text, encoding="utf-8")

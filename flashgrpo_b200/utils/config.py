from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    parent = data.pop("inherits", None)
    if parent:
        parent_path = (path.parent / parent).resolve()
        return deep_update(load_config(parent_path), data)
    return data


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def set_by_dotted_key(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    cur = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def parse_override(text: str) -> tuple[str, Any]:
    key, raw = text.split("=", 1)
    try:
        value = yaml.safe_load(raw)
    except Exception:
        value = raw
    return key, value


def save_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def to_pretty_json(config: dict[str, Any]) -> str:
    return json.dumps(config, indent=2, ensure_ascii=True)

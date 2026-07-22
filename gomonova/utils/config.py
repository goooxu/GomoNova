"""YAML config loader with _base_ inheritance."""

from __future__ import annotations

import os

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if "_base_" in cfg:
        base_name = cfg.pop("_base_")
        base_path = os.path.join(os.path.dirname(path), base_name)
        base_cfg = load_config(base_path)
        cfg = _deep_merge(base_cfg, cfg)

    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result

from functools import lru_cache
from pathlib import Path

import yaml


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "config.yaml"


@lru_cache(maxsize=1)
def load_model_config() -> dict:
    """Load shared model and dashboard parameters from the root config file."""
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    required_keys = {
        "media_cols",
        "channel_labels",
        "hill_params",
        "sat_cols",
        "control_cols",
        "default_weekly_budget",
    }
    missing = required_keys - set(config)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise KeyError(f"Missing required config key(s): {missing_list}")

    return config

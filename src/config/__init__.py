"""Configuration module."""

from pathlib import Path
import yaml

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load configuration
CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "config.yaml"


def load_config() -> dict:
    """Load configuration from YAML file."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    return {}


config = load_config()


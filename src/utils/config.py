from pathlib import Path
from typing import Type, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def load_config(config_path: Path, config_schema: Type[T]) -> T:
    """
    Load and validates the approach configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.
        config_schema: The Pydantic model class to validate against.

    Returns:
        A validated ApproachConfig object.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the configuration is invalid.

    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r") as f:
        try:
            config_data = yaml.safe_load(f)
            return config_schema(**config_data)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")
        except ValidationError as e:
            raise ValueError(f"Configuration validation error: {e}")
        except Exception as e:
            raise ValueError(f"An unexpected error occurred while loading config: {e}")

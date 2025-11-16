import argparse
import importlib
import sys
from pathlib import Path
from typing import List


def main() -> None:
    """Execute from command-line."""
    parser = argparse.ArgumentParser(
        description=(
            "ML Project Runner CLI. "
            "For interactive mode, run 'python main_interactive.py'"
        ),
    )
    parser.add_argument(
        "-a",
        "--approach",
        type=str,
        required=True,
        choices=get_approaches(),
        help="The name of the approach to run.",
    )
    parser.add_argument(
        "-x",
        "--action",
        type=str,
        required=True,
        choices=["train", "validate"],
        help="The action to perform.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="The name of the configuration file (without .yaml extension).",
    )
    args = parser.parse_args()

    # Validate that the chosen config exists for the chosen approach
    if args.config not in get_configs(args.approach):
        print(
            f"Error: Config '{args.config}' not found for approach '{args.approach}'.\n"
            f"Available configs: {get_configs(args.approach)}",
        )
        sys.exit(1)

    run_action(args.approach, args.action, args.config)


def run_action(approach: str, action: str, config: str) -> None:
    """Dynamically import and run the specified action from the given approach."""
    print(
        f"\nRunning '{action}' for approach '{approach}' with config",
        f"'{config}.yaml'...",
    )
    print("-" * 30)

    try:
        module_path = f"src.approaches.{approach}.{action}"
        action_module = importlib.import_module(module_path)
        action_function = getattr(action_module, action)
        action_function(config_name=config)

    except ModuleNotFoundError:
        print(
            f"Error: Could not find a '{action}.py' file for the '{approach}'",
            "approach.",
        )
        sys.exit(1)
    except AttributeError:
        print(f"Error: The '{action}.py' file does not have a '{action}()' function.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

    print("-" * 30)
    print("Run completed successfully.")


def get_approaches() -> List[str]:
    """Find all available approaches in the src/approaches directory."""
    approaches_dir = Path("src") / "approaches"
    if not approaches_dir.exists():
        return []
    return [
        d.name
        for d in approaches_dir.iterdir()
        if d.is_dir() and not d.name.startswith("__")
    ]


def get_configs(approach: str) -> List[str]:
    """Find all YAML config files for a given approach."""
    configs_dir = Path("src") / "approaches" / approach / "configs"
    if not configs_dir.exists():
        return []
    return [f.stem for f in configs_dir.glob("*.yaml")]


if __name__ == "__main__":
    main()


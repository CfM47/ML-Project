from typing import List, Optional

# Import the core logic helpers from main.py
from main import get_approaches, get_configs, run_action


def select_choice(options: List[str], prompt: str) -> Optional[str]:
    """Display a numbered list of options and get the user's choice."""
    if not options:
        print(f"No options found for: {prompt}")
        return None

    print(prompt)
    for i, option in enumerate(options):
        print(f"  {i + 1}: {option}")

    while True:
        try:
            choice = input(f"Enter a number (1-{len(options)}): ")
            choice_index = int(choice) - 1
            if 0 <= choice_index < len(options):
                return options[choice_index]
            else:
                print("Invalid number, please try again.")
        except ValueError:
            print("Invalid input, please enter a number.")


def main() -> None:
    """Run the TUI for interactively selecting and running an experiment."""
    print("--- ML Project Runner (Interactive Mode) ---")

    # 1. Select an approach
    approaches = get_approaches()
    selected_approach = select_choice(approaches, "Select an approach:")
    if not selected_approach:
        return

    # 2. Select an action
    actions = ["train", "validate"]
    selected_action = select_choice(actions, "Select an action:")
    if not selected_action:
        return

    # 3. Select a config
    configs = get_configs(selected_approach)
    selected_config = select_choice(configs, "Select a configuration:")
    if not selected_config:
        return

    # 4. Run the selected action using the helper from main.py
    run_action(selected_approach, selected_action, selected_config)


if __name__ == "__main__":
    main()

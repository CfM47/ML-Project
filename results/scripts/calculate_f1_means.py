"""Calculate mean F1 scores for each model and augmentator."""

import json
from pathlib import Path
from typing import Any

# Directory where this script lives
SCRIPT_DIR = Path(__file__).parent
# Results directory is the parent of scripts/
RESULTS_DIR = SCRIPT_DIR.parent


def calculate_f1_means(results: dict[str, Any]) -> dict[str, dict[str, float]]:
    """
    Calculate the mean F1_Macro score for each model and augmentator.

    Returns a dictionary with structure:
    {
        "augmentator_name": {
            "model_name": mean_f1_score,
            ...
        },
        ...
    }
    """
    f1_means: dict[str, dict[str, float]] = {}

    for experiment_key, experiment_data in results.items():
        f1_means[experiment_key] = {}

        for model_key, model_data in experiment_data.items():
            if "evaluations" not in model_data:
                continue

            evaluations = model_data["evaluations"]

            data = {}

            # Calculate mean F1_Macro if it exists
            if "F1_Macro" in evaluations:
                f1_values = evaluations["F1_Macro"]
                mean_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0
                data["mean_f1"] = mean_f1
            if "Accuracy" in evaluations:
                acc_values = evaluations["Accuracy"]
                mean_acc = sum(acc_values) / len(acc_values) if acc_values else 0.0
                data["mean_accuracy"] = mean_acc
            if "Mask_Cohesion" in evaluations:
                mask_values = evaluations["Mask_Cohesion"]
                mean_mask = sum(mask_values) / len(mask_values) if mask_values else 0.0
                data["mean_mask_cohesion"] = mean_mask

            f1_means[experiment_key][model_key] = data
    return f1_means


def main() -> None:
    """Run the F1 mean calculation."""
    results_path = RESULTS_DIR / "results_cache.json"
    output_path = RESULTS_DIR / "f1_means.json"

    if not results_path.exists():
        print(f"Error: {results_path} does not exist")
        return

    # Load the results
    print(f"Loading results from {results_path}...")
    with open(results_path) as f:
        results = json.load(f)

    # Calculate F1 means
    print("Calculating mean F1 scores for each model and augmentator...")
    f1_means = calculate_f1_means(results)

    # Save the F1 means
    print(f"Saving F1 means to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(f1_means, f, indent=2)

    print("Done! F1 means have been saved.")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()

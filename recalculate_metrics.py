"""Recalculate macro metrics in the results cache file."""

import json
from pathlib import Path
from typing import Any


def recalculate_macro_metrics(results: dict[str, Any]) -> dict[str, Any]:
    """
    Recalculate macro metrics as the average of Class0 and Class1 only.

    Also calculates F1_Macro using the recalculated Precision_Macro and Recall_Macro.
    """
    for experiment_key, experiment_data in results.items():
        for model_key, model_data in experiment_data.items():
            if "evaluations" not in model_data:
                continue

            evaluations = model_data["evaluations"]

            # Recalculate IoU_Macro
            if "IoU_Class0" in evaluations and "IoU_Class1" in evaluations:
                class0 = evaluations["IoU_Class0"]
                class1 = evaluations["IoU_Class1"]
                evaluations["IoU_Macro"] = [
                    (c0 + c1) / 2 for c0, c1 in zip(class0, class1)
                ]

            # Recalculate Dice_Macro
            if "Dice_Class0" in evaluations and "Dice_Class1" in evaluations:
                class0 = evaluations["Dice_Class0"]
                class1 = evaluations["Dice_Class1"]
                evaluations["Dice_Macro"] = [
                    (c0 + c1) / 2 for c0, c1 in zip(class0, class1)
                ]

            # Recalculate Precision_Macro
            if "Precision_Class0" in evaluations and "Precision_Class1" in evaluations:
                class0 = evaluations["Precision_Class0"]
                class1 = evaluations["Precision_Class1"]
                evaluations["Precision_Macro"] = [
                    (c0 + c1) / 2 for c0, c1 in zip(class0, class1)
                ]

            # Recalculate Recall_Macro
            if "Recall_Class0" in evaluations and "Recall_Class1" in evaluations:
                class0 = evaluations["Recall_Class0"]
                class1 = evaluations["Recall_Class1"]
                evaluations["Recall_Macro"] = [
                    (c0 + c1) / 2 for c0, c1 in zip(class0, class1)
                ]

            # Calculate F1_Macro using Precision_Macro and Recall_Macro
            if "Precision_Macro" in evaluations and "Recall_Macro" in evaluations:
                precision = evaluations["Precision_Macro"]
                recall = evaluations["Recall_Macro"]
                evaluations["F1_Macro"] = [
                    (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
                    for p, r in zip(precision, recall)
                ]

    return results


def main() -> None:
    """Run the recalculation of macro metrics."""
    results_path = Path("results/results_cache.json")

    if not results_path.exists():
        print(f"Error: {results_path} does not exist")
        return

    # Load the results
    print(f"Loading results from {results_path}...")
    with open(results_path, "r") as f:
        results = json.load(f)

    # Recalculate macro metrics
    print("Recalculating macro metrics...")
    results = recalculate_macro_metrics(results)

    # Save the updated results
    print(f"Saving updated results to {results_path}...")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Done! Macro metrics have been recalculated.")
    print(
        "- IoU_Macro, Dice_Macro, Precision_Macro, Recall_Macro: "
        "average of Class0 and Class1",
    )
    print(
        "- F1_Macro: calculated as "
        "2 * (Precision_Macro * Recall_Macro) / (Precision_Macro + Recall_Macro)",
    )


if __name__ == "__main__":
    main()

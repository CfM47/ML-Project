import copy
from typing import Any, Dict, List, Optional

from auto_ml.implementations import DataAugmentatorNode, EvaluatorNode, ModelNode
from auto_ml.interfaces import DatasetInterface, MaskPair


class AutoML:
    """
    AutoML Orchestrator.

    Manages the execution of experiments across multiple data augmentation
    strategies and models.
    """

    def __init__(self) -> None:  # noqa: D107
        self.results: Dict[str, Any] = {}

    def run_experiment(
        self,
        dataset: DatasetInterface,
        augmentator_nodes: List[DataAugmentatorNode],
        model_nodes: List[ModelNode],
        evaluator_node: Optional[EvaluatorNode] = None,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Run a full experiment.

        Iterates over each augmentator node and each model node.

        Args:
            dataset: The base dataset.
            augmentator_nodes: List of DataAugmentatorNode instances.
            model_nodes: List of ModelNode instances.
            evaluator_node: Optional single EvaluatorNode instance.

        Returns:
            Dictionary structure:
            {
                augmentator_name: {
                    model_name: {
                        "mask_pairs": List[List[MaskPair]],
                        "evaluation": Dict[str, Any] (if evaluator_node provided)
                    }
                }
            }

        """
        print(
            f"Starting AutoML Experiment with {len(augmentator_nodes)} augmentators",
            f"and {len(model_nodes)} models.",
        )

        experiment_results: Dict[str, Any] = {}

        for aug_node in augmentator_nodes:
            aug_name = aug_node.name
            print(f"\nProcessing Data Augmentation Node: {aug_name}")
            experiment_results[aug_name] = {}

            # 1. Process Data (Split & Augment)
            try:
                dataset_pairs = aug_node.process(dataset)
                print(f"  Generated {len(dataset_pairs)} dataset pairs (folds).")

                # 2. Iterate Models
                for model_node_template in model_nodes:
                    model_name = model_node_template.name
                    print(f"  Training Model Node: {model_name}")

                    try:
                        # CRITICAL: Use a fresh copy of the model for this pipeline run
                        # to ensure we start training from scratch (0).
                        model_node = copy.deepcopy(model_node_template)

                        # Train Model - returns List[List[MaskPair]]
                        mask_pairs = model_node.train(dataset_pairs)

                        total_pairs = sum(len(fold) for fold in mask_pairs)
                        print(f"    Finished. Collected {total_pairs} mask pairs.")

                        # Build result dict
                        result: Dict[str, Any] = {
                            "mask_pairs": mask_pairs,
                        }

                        # 3. Pass to Evaluator Node (if provided)
                        if evaluator_node:
                            evaluation_results = evaluator_node.evaluate(mask_pairs)
                            result["evaluation"] = evaluation_results

                        experiment_results[aug_name][model_name] = result

                    except Exception as e:
                        print(f"    Error training {model_name} on {aug_name}: {e}")
                        experiment_results[aug_name][model_name] = {"error": str(e)}

            except Exception as e:
                print(f"  Error processing augmentation {aug_name}: {e}")
                experiment_results[aug_name] = {"error": str(e)}

        self.results = experiment_results
        return experiment_results

    def get_summary(self) -> str:
        """
        Get a readable summary of the experiment results.

        Returns:
            String summary.

        """
        if not self.results:
            return "No results available."

        summary = "=== AutoML Experiment Summary ===\n"

        for aug_name, models_data in self.results.items():
            summary += f"\nData Augmentation: {aug_name}\n"

            if "error" in models_data:
                summary += f"  Error: {models_data['error']}\n"
                continue

            for model_name, res in models_data.items():
                if isinstance(res, dict) and "error" in res:
                    summary += f"  Model: {model_name} -> Error: {res['error']}\n"
                elif isinstance(res, dict):
                    mask_pairs = res.get("mask_pairs", [])
                    total_pairs = sum(len(fold) for fold in mask_pairs)
                    summary += f"  Model: {model_name} -> {len(mask_pairs)} folds, "
                    summary += f"{total_pairs} mask pairs\n"

                    # Show evaluation results if available
                    if "evaluation" in res:
                        for eval_name, eval_result in res["evaluation"].items():
                            summary += f"    {eval_name}: {eval_result}\n"

        summary += "\n================================="
        return summary


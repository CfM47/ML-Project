import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from auto_ml.implementations import DataAugmentatorNode, EvaluatorNode, ModelNode
from auto_ml.interfaces import SegmentationDatasetInterface


class AutoML:
    """
    AutoML Orchestrator.

    Manages the execution of experiments across multiple data augmentation
    strategies and models. Includes caching and execution time tracking.
    """

    def __init__(self, cache_dir: str = "automl_cache") -> None:  # noqa: D107
        self.results: Dict[str, Any] = {}
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.results_cache_path = self.cache_dir / "results_cache.json"
        self.execution_times_cache_path = self.cache_dir / "execution_times_cache.json"

        self.results_cache: Dict[str, Any] = {}
        self.execution_times_cache: Dict[str, Any] = {}

        self._load_caches()

    def _load_caches(self) -> None:
        """Load results and execution times caches from JSON files."""
        if self.results_cache_path.exists():
            try:
                with open(self.results_cache_path, "r") as f:
                    self.results_cache = json.load(f)
                print(f"Loaded results cache from {self.results_cache_path}")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load results cache: {e}")
                self.results_cache = {}

        if self.execution_times_cache_path.exists():
            try:
                with open(self.execution_times_cache_path, "r") as f:
                    self.execution_times_cache = json.load(f)
                print(
                    f"Loaded execution times cache from "
                    f"{self.execution_times_cache_path}",
                )
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load execution times cache: {e}")
                self.execution_times_cache = {}

    def _save_results_cache(self) -> None:
        """Save results cache to JSON file."""
        try:
            with open(self.results_cache_path, "w") as f:
                json.dump(self.results_cache, f, indent=2, default=str)
            print(f"Saved results cache to {self.results_cache_path}")
        except IOError as e:
            print(f"Warning: Failed to save results cache: {e}")

    def _save_execution_times_cache(self) -> None:
        """Save execution times cache to JSON file."""
        try:
            with open(self.execution_times_cache_path, "w") as f:
                json.dump(self.execution_times_cache, f, indent=2)
            print(
                f"Saved execution times cache to {self.execution_times_cache_path}",
            )
        except IOError as e:
            print(f"Warning: Failed to save execution times cache: {e}")

    def _clear_cache_entry(self, augmentator_name: str, model_name: str) -> None:
        """Clear cache entry for a specific augmentator+model pair."""
        if augmentator_name in self.results_cache:
            if model_name in self.results_cache[augmentator_name]:
                del self.results_cache[augmentator_name][model_name]

        if augmentator_name in self.execution_times_cache:
            if model_name in self.execution_times_cache[augmentator_name]:
                del self.execution_times_cache[augmentator_name][model_name]

        self._save_results_cache()
        self._save_execution_times_cache()
        print(f"Cleared cache for {augmentator_name}:{model_name}")

    def _is_cached(self, augmentator_name: str, model_name: str) -> bool:
        """Check if a combination is in the results cache."""
        return (
            augmentator_name in self.results_cache
            and model_name in self.results_cache[augmentator_name]
        )

    def _get_cached_result(
        self, augmentator_name: str, model_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached result for a specific combination."""
        if self._is_cached(augmentator_name, model_name):
            result: Dict[str, Any] = self.results_cache[augmentator_name][
                model_name
            ]
            return result
        return None

    def _cache_result(
        self,
        augmentator_name: str,
        model_name: str,
        result: Dict[str, Any],
        execution_time: float,
    ) -> None:
        """Cache result and execution time for a specific combination."""
        if augmentator_name not in self.results_cache:
            self.results_cache[augmentator_name] = {}

        self.results_cache[augmentator_name][model_name] = result

        if augmentator_name not in self.execution_times_cache:
            self.execution_times_cache[augmentator_name] = {}

        self.execution_times_cache[augmentator_name][model_name] = execution_time

        self._save_results_cache()
        self._save_execution_times_cache()

    def run_experiment(
        self,
        dataset: SegmentationDatasetInterface,
        augmentator_nodes: List[DataAugmentatorNode],
        model_nodes: List[ModelNode],
        evaluator_node: Optional[EvaluatorNode] = None,
        clear_cache: Optional[List[tuple]] = None,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Run a full experiment.

        Iterates over each augmentator node and each model node.

        Args:
            dataset: The base dataset.
            augmentator_nodes: List of DataAugmentatorNode instances.
            model_nodes: List of ModelNode instances.
            evaluator_node: Optional single EvaluatorNode instance.
            clear_cache: Optional list of (augmentator_name, model_name)
                tuples to clear.

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
        # Clear cache entries if specified
        if clear_cache:
            for aug_name, model_name in clear_cache:
                self._clear_cache_entry(aug_name, model_name)

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

                    # Check cache first
                    if self._is_cached(aug_name, model_name):
                        print(f"  Training Model Node: {model_name} (cached)")
                        cached_result = self._get_cached_result(
                            aug_name, model_name,
                        )
                        execution_time = (
                            self.execution_times_cache[aug_name][model_name]
                        )
                        experiment_results[aug_name][model_name] = cached_result
                        print(
                            f"    Loaded from cache "
                            f"(execution time: {execution_time:.2f}s)",
                        )
                        continue

                    print(f"  Training Model Node: {model_name}")
                    cycle_start_time = time.time()

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
                            "train_size": len(dataset_pairs[0][0]),
                            "test_size": len(dataset_pairs[0][1]),
                        }

                        # 3. Pass to Evaluator Node (if provided)
                        if evaluator_node:
                            evaluation_results = evaluator_node.evaluate(mask_pairs)
                            result["evaluations"] = evaluation_results

                        experiment_results[aug_name][model_name] = result

                        # Cache the result and execution time
                        cycle_end_time = time.time()
                        execution_time = cycle_end_time - cycle_start_time

                        # Extract training history if available
                        # Extract training history if available
                        if (
                            hasattr(model_node, "last_training_metrics")
                            and model_node.last_training_metrics
                        ):
                            # We might have multiple folds.
                            # Let's simple store the history of the last fold
                            # or list of all.
                            # Since result is per model/aug pair,
                            # maybe we want to store all?
                            # For simplicity, append the history to the result object
                            # We'll use the 'history' key which is a list of lists
                            # (one per fold)
                            result["training_history"] = [
                                m.history for m in model_node.last_training_metrics
                            ]

                        self._cache_result(
                            aug_name, model_name, result, execution_time,
                        )
                        print(
                            f"    Cached result "
                            f"(execution time: {execution_time:.2f}s)",
                        )

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


from typing import List, Dict, Any
import copy
from auto_ml.interfaces import DatasetInterface
from auto_ml.implementations import DataAugmentatorNode, ModelNode

class AutoML:
    """
    AutoML Orchestrator.
    
    Manages the execution of experiments across multiple data augmentation
    strategies and models.
    """
    
    def __init__(self):
        self.results = {}

    def run_experiment(
        self, 
        dataset: DatasetInterface, 
        augmentator_nodes: List[DataAugmentatorNode], 
        model_nodes: List[ModelNode]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run a full experiment.
        
        Iterates over each augmentator node and each model node.
        
        Args:
            dataset: The base dataset.
            augmentator_nodes: List of DataAugmentatorNode instances.
            model_nodes: List of ModelNode instances.
            
        Returns:
            Dictionary structure:
            {
                augmentator_name: {
                    model_name: result_dict
                }
            }
        """
        print(f"Starting AutoML Experiment with {len(augmentator_nodes)} augmentators and {len(model_nodes)} models.")
        
        experiment_results = {}
        
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
                        
                        # Train Model
                        result = model_node.train(dataset_pairs)
                        experiment_results[aug_name][model_name] = result
                        print(f"    Finished. Mean Loss: {result['mean_loss']:.4f}")
                        
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
                if "error" in res:
                    summary += f"  Model: {model_name} -> Error: {res['error']}\n"
                else:
                    mean_loss = res.get("mean_loss", "N/A")
                    # Try to get mean accuracy if available in results list
                    mean_acc = "N/A"
                    if "results" in res and res["results"]:
                         accs = [r.accuracy for r in res["results"]]
                         mean_acc = sum(accs) / len(accs)
                         
                    summary += f"  Model: {model_name} -> Mean Loss: {mean_loss}, Mean Acc: {mean_acc}\n"
                    
        summary += "\n================================="
        return summary

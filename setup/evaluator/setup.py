from auto_ml.implementations.evaluators import (
    AccuracyEvaluator,
    AutoencoderMaskEvaluator,
    DiceClass0Evaluator,
    DiceClass1Evaluator,
    DiceClass2Evaluator,
    DiceMacroAverageEvaluator,
    DiceWeightedAverageEvaluator,
    IoUClass0Evaluator,
    IoUClass1Evaluator,
    IoUClass2Evaluator,
    IoUMacroAverageEvaluator,
    IoUWeightedAverageEvaluator,
    PrecisionClass0Evaluator,
    PrecisionClass1Evaluator,
    PrecisionClass2Evaluator,
    PrecisionMacroAverageEvaluator,
    RecallClass0Evaluator,
    RecallClass1Evaluator,
    RecallClass2Evaluator,
    RecallMacroAverageEvaluator,
)
from auto_ml.implementations.nodes import EvaluatorNode
from auto_ml.interfaces import SegmentationDatasetInterface


def get_evaluator_node(dataset: SegmentationDatasetInterface) -> EvaluatorNode:
    """Return evaluator node for use in Auto-ML."""
    evaluators = {
        # General
        "Accuracy": AccuracyEvaluator(),

        # Autoencoder (requires training on reference masks)
        "Mask_Cohesion": AutoencoderMaskEvaluator(
            reference_masks=dataset.masks,
            latent_dim=8,
            epochs=40,
            nu=0.5,
            device="auto",
        ),

        # IoU Metrics
        "IoU_Class0": IoUClass0Evaluator(),
        "IoU_Class1": IoUClass1Evaluator(),
        "IoU_Class2": IoUClass2Evaluator(),
        "IoU_Macro": IoUMacroAverageEvaluator(),
        "IoU_Weighted": IoUWeightedAverageEvaluator(),

        # Dice Metrics
        "Dice_Class0": DiceClass0Evaluator(),
        "Dice_Class1": DiceClass1Evaluator(),
        "Dice_Class2": DiceClass2Evaluator(),
        "Dice_Macro": DiceMacroAverageEvaluator(),
        "Dice_Weighted": DiceWeightedAverageEvaluator(),

        # Precision Metrics
        "Precision_Class0": PrecisionClass0Evaluator(),
        "Precision_Class1": PrecisionClass1Evaluator(),
        "Precision_Class2": PrecisionClass2Evaluator(),
        "Precision_Macro": PrecisionMacroAverageEvaluator(),

        # Recall Metrics
        "Recall_Class0": RecallClass0Evaluator(),
        "Recall_Class1": RecallClass1Evaluator(),
        "Recall_Class2": RecallClass2Evaluator(),
        "Recall_Macro": RecallMacroAverageEvaluator(),
    }
    return EvaluatorNode(evaluators=evaluators)

from typing import List

from auto_ml.implementations.nodes import DataAugmentatorNode
from setup.augmentators.baseline import get_baseline_node
from setup.augmentators.combined_2geo_2photo_1sem import (
    get_combined_2geo_2photo_1sem_node,
)
from setup.augmentators.combined_3geo_1photo_1sem import (
    get_combined_3geo_1photo_1sem_node,
)


def get_augmentator_nodes(
    baseline_copies: int = 0,
    combined_2geo_2photo_1sem_copies: int = 2,
    combined_3geo_1photo_1sem_copies: int = 2,
) -> List[DataAugmentatorNode]:
    """
    Return a list of augmentator nodes with efficient combinations for optimal results.

    Args:
        baseline_copies: Number of copies for baseline node (default: 0).
        combined_2geo_2photo_1sem_copies: Number of copies for 2geo+2photo+1sem
            (default: 2).
        combined_3geo_1photo_1sem_copies: Number of copies for 3geo+1photo+1sem
            (default: 2).

    """
    return [
        get_baseline_node(num_copies=baseline_copies),
        get_combined_2geo_2photo_1sem_node(num_copies=combined_2geo_2photo_1sem_copies),
        get_combined_3geo_1photo_1sem_node(num_copies=combined_3geo_1photo_1sem_copies),
    ]

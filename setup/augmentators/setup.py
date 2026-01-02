from typing import List

from auto_ml.implementations.nodes import DataAugmentatorNode
from setup.augmentators.baseline import get_baseline_node
from setup.augmentators.combined_1geo_1photo_1sem import (
    get_combined_1geo_1photo_1sem_node,
)
from setup.augmentators.combined_2geo_1photo_1sem import (
    get_combined_2geo_1photo_1sem_node,
)
from setup.augmentators.combined_2geo_1sem import get_combined_2geo_1sem_node
from setup.augmentators.combined_2photo_1geo import get_combined_2photo_1geo_node


def get_augmentator_nodes(
    baseline_copies: int = 1,
    combined_2geo_1photo_1sem_copies: int = 3,
    combined_2geo_1sem_copies: int = 5,
    combined_2photo_1geo_copies: int = 4,
    combined_1geo_1photo_1sem_copies: int = 3,
) -> List[DataAugmentatorNode]:
    """Return a list of augmentator nodes with efficient combinations for optimal results.

    Args:
        baseline_copies: Number of copies for baseline node (default: 1).
        combined_2geo_1photo_1sem_copies: Number of copies for 2geo+1photo+1sem (default: 1).
        combined_2geo_1sem_copies: Number of copies for 2geo+1sem (default: 1).
        combined_2photo_1geo_copies: Number of copies for 2photo+1geo (default: 1).
        combined_1geo_1photo_1sem_copies: Number of copies for 1geo+1photo+1sem (default: 1).

    """
    return [
        get_baseline_node(num_copies=baseline_copies),
        get_combined_2geo_1photo_1sem_node(num_copies=combined_2geo_1photo_1sem_copies),
        get_combined_2geo_1sem_node(num_copies=combined_2geo_1sem_copies),
        get_combined_2photo_1geo_node(num_copies=combined_2photo_1geo_copies),
        get_combined_1geo_1photo_1sem_node(num_copies=combined_1geo_1photo_1sem_copies),
    ]

"""Hardware-aware layout and feature-assignment utilities."""

from pqfmlib.hardware.feature_assignment import (
    GA_assignment,
    GA_assignment_multiaxis,
    pick_connected_subset_greedy,
)

__all__ = ["GA_assignment", "GA_assignment_multiaxis", "pick_connected_subset_greedy"]

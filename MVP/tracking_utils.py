import numpy as np
from MVP.data_structures.match import Match
from scipy.optimize import linear_sum_assignment
from typing import (
    List,
    TypeVar,
    Callable,
    Tuple,
    Set,
)
from copy import deepcopy

T = TypeVar("T")


def get_hungarian_algorithm_matches(
    predictions: List[T],
    ground_true: List[T],
    weight_measurer: Callable[[T, T], float],
) -> Tuple[List[Match[int]], np.ndarray]:
    """Performs hungariam algorithm matching, and returns matches of indexes and weights matrix"""
    weights_matrix = np.zeros((len(ground_true), len(predictions)))
    for gt_ind, gt_obj in enumerate(ground_true):
        for pred_ind, pred_obj in enumerate(predictions):
            weight: float = weight_measurer(pred_obj, gt_obj)
            weights_matrix[gt_ind][pred_ind] = weight
    gt_ixs, face_ixs = linear_sum_assignment(weights_matrix)
    matches: List[Match[int]] = [
        Match(gt_value=gt_val, pred_value=pred_val)
        for gt_val, pred_val in zip(gt_ixs, face_ixs)
    ]
    return matches, weights_matrix


def get_fp_instances(predictions: List[T], matches: List[Match[int]]) -> List[T]:
    """Returns predictions instances that were not matched to any gt, match is represented as a pair of indexes"""
    all_predictions_indexes: Set[int] = set(range(len(predictions)))
    matched_indexes: Set[int] = set([m.pred_value for m in matches])
    unmatched_indexes: Set[int] = all_predictions_indexes - matched_indexes
    fp_instances: List[T] = [predictions[i] for i in unmatched_indexes]
    return fp_instances


def get_fn_instances(gt_instances: List[T], matches: List[Match[int]]) -> List[T]:
    """Returns gt instances that were not matched to any prediction, match is represented as a pair of indexes"""
    all_gt_indexes: Set[int] = set(range(len(gt_instances)))
    matched_indexes: Set[int] = set([m.gt_value for m in matches])
    unmatched_indexes: Set[int] = all_gt_indexes - matched_indexes
    fn_instances: List[T] = [gt_instances[i] for i in unmatched_indexes]
    return fn_instances

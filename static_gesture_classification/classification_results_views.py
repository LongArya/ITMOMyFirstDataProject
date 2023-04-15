from static_gesture_classification.classification_results_dataframe import (
    ClassificationResultsDataframe,
)
from static_gesture_classification.static_gesture import StaticGesture


def get_gt_pred_combination_view(
    dataframe_predictions: ClassificationResultsDataframe,
    gt_gesture: StaticGesture,
    pred_gesture: StaticGesture,
) -> ClassificationResultsDataframe:
    """Returns view with specific ground true and prediction"""
    gt_pred_mask = dataframe_predictions.apply(
        lambda row: row.ground_true == gt_gesture.name
        and row.prediction == pred_gesture.name,
        axis=1,
    )
    fails_view: ClassificationResultsDataframe = dataframe_predictions[gt_pred_mask]
    return fails_view


def get_fails_view(
    dataframe_predictions: ClassificationResultsDataframe,
) -> ClassificationResultsDataframe:
    """Return view with all fails, a.i. rows where prediction and ground true do not match"""
    fails_mask = dataframe_predictions.apply(
        lambda row: row.ground_true != row.prediction,
        axis=1,
    )
    fails_view: ClassificationResultsDataframe = dataframe_predictions[fails_mask]
    return fails_view


def get_gesture_view(
    classification_results: ClassificationResultsDataframe,
    target_gesture: StaticGesture,
) -> ClassificationResultsDataframe:
    """Return only part where either prediction or ground true is target gesture"""
    mask = classification_results.apply(
        lambda row: row.ground_true == target_gesture.name
        or row.prediction == target_gesture.name,
        axis=1,
    )
    view = classification_results[mask]
    return view

import torch
from torchmetrics.classification import BinaryPrecisionRecallCurve
from matplotlib.axes._axes import Axes
import matplotlib.pyplot as plt
from static_gesture_classification.classification_results_dataframe import (
    ClassificationResultsDataframe,
)
from sklearn.metrics import confusion_matrix, precision_recall_curve
import numpy as np
import pandas as pd
from static_gesture_classification.static_gesture import StaticGesture
import seaborn as sns
from dataclasses import dataclass
from typing import Iterable, List, Dict


@dataclass
class PRCurve:
    """Container for precision recall curve description"""

    precision_values: Iterable[float]
    recall_values: Iterable[float]
    thresholds: Iterable[float]


def generate_confusion_matrix_plot_from_classification_results(
    prediction_results: ClassificationResultsDataframe, plot_axis: Axes
) -> Axes:
    """Plots confusion matrix on given axis based on provided prediction results"""
    ground_true = prediction_results.ground_true.tolist()
    predictions = prediction_results.prediction.tolist()
    labels = [gesture.name for gesture in StaticGesture]
    conf_mat: np.ndarray = confusion_matrix(
        y_true=ground_true,
        y_pred=predictions,
        labels=labels,
    )
    conf_mat_dataframe: pd.DataFrame = pd.DataFrame(
        data=conf_mat, index=labels, columns=labels
    )
    plot_axis = sns.heatmap(data=conf_mat_dataframe, ax=plot_axis, annot=True)
    return plot_axis


def get_pr_curve_plot(plot_axis: Axes, pr_curve: PRCurve) -> Axes:
    plot_axis.plot(pr_curve.recall_values, pr_curve.precision_values)
    plot_axis.set_xlabel("Recall")
    plot_axis.set_ylabel("Precision")
    plot_axis.set_xlim([-0.1, 1.1])
    plot_axis.set_ylim([-0.1, 1.1])
    plot_axis.set_aspect(aspect="equal")
    return plot_axis


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


def compute_pr_curve_for_gesture(
    results: ClassificationResultsDataframe, target_gesture: StaticGesture
) -> PRCurve:
    # get gesture view
    gesture_view = get_gesture_view(
        classification_results=results, target_gesture=target_gesture
    )
    gesture_scores = gesture_view[target_gesture.name].tolist()
    ground_true = gesture_view.ground_true.tolist()
    binarized_ground_true = [
        1 if gesture == target_gesture.name else 0 for gesture in ground_true
    ]
    precision, recall, thresholds = precision_recall_curve(
        y_true=binarized_ground_true, probas_pred=gesture_scores
    )
    pr_curve = PRCurve(
        precision_values=precision, recall_values=recall, thresholds=thresholds
    )
    return pr_curve


def get_pr_curves_for_gestures(
    classification_results: ClassificationResultsDataframe,
) -> Dict[StaticGesture, PRCurve]:
    pr_curves = {}
    for gesture in StaticGesture:
        if gesture == StaticGesture.BACKGROUND:
            continue
        gesture_pr_curve = compute_pr_curve_for_gesture(classification_results, gesture)
        pr_curves[gesture] = gesture_pr_curve
    return pr_curves

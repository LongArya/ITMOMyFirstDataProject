import torch
from typing import Tuple
import matplotlib.pyplot as plt
import onnxruntime
import numpy as np
import cv2
from static_gesture_classification.static_gesture import StaticGesture
from scipy.special import softmax


class ONNXResnet18StaticGestureClassifier:
    """Class for encapsulating inference of resnet18 trained for static gesture classification via ONNX"""

    def __init__(self, model_path: str):
        self.ort_sess = onnxruntime.InferenceSession(
            model_path, providers=["CUDAExecutionProvider"]
        )
        self._input_size = (224, 224)
        self._imagenet_mean = np.array([0.485, 0.456, 0.406])
        self._imagenet_std = np.array([0.229, 0.224, 0.225])

    def _preprocessing(self, input_image: np.ndarray) -> np.ndarray:
        network_input = input_image.astype(np.float32)
        network_input = cv2.resize(network_input, self._input_size)
        network_input /= 255
        # tensor.sub_(mean).div_(std)
        network_input -= self._imagenet_mean
        network_input /= self._imagenet_std
        network_input = np.transpose(network_input, (2, 0, 1))
        network_input = np.expand_dims(network_input, 0)
        return network_input

    def _parse_model_output(
        self, model_logits: np.ndarray
    ) -> Tuple[StaticGesture, float]:
        probs = softmax(model_logits)
        pred_class_index = np.argmax(probs)
        prediction_probability: float = probs[pred_class_index]
        predicted_gesture: StaticGesture = StaticGesture(pred_class_index)
        return predicted_gesture, prediction_probability

    def __call__(self, image: np.ndarray) -> Tuple[StaticGesture, float]:
        network_input = self._preprocessing(image)
        model_logits = self.ort_sess.run(None, {"input.1": network_input})[0][0]
        predicted_gesture, prediction_probability = self._parse_model_output(
            model_logits
        )
        return predicted_gesture, prediction_probability

from typing import Tuple
import onnxruntime
import numpy as np
import cv2


def letterbox(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
):
    """Implementation of letterbox transformation, image is resized without changing aspect ratio,
    and padded in order to gain the desired output resolution"""
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    # dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return img, ratio, (dw, dh)


def scale_boxes_to_original_image_space(
    input_img_hw_shape: Tuple[int, int],
    xyxy_boxes: np.ndarray,
    orig_image_hw_shape: Tuple[int, int],
) -> np.ndarray:
    """Scales boxes from coordinates of letterboxed image to original image coordinates"""
    gain = min(
        input_img_hw_shape[0] / orig_image_hw_shape[0],
        input_img_hw_shape[1] / orig_image_hw_shape[1],
    )
    pad = (input_img_hw_shape[1] - orig_image_hw_shape[1] * gain) / 2, (
        input_img_hw_shape[0] - orig_image_hw_shape[0] * gain
    ) / 2

    xyxy_boxes[:, [0, 2]] -= pad[0]  # x padding
    xyxy_boxes[:, [1, 3]] -= pad[1]  # y padding
    xyxy_boxes[:, :4] /= gain
    clip_coords(xyxy_boxes, orig_image_hw_shape)
    return xyxy_boxes


def clip_coords(xyxy_boxes, img_shape_hw):
    xyxy_boxes[:, 0] = np.clip(xyxy_boxes[:, 0], 0, img_shape_hw[1])
    xyxy_boxes[:, 1] = np.clip(xyxy_boxes[:, 1], 0, img_shape_hw[0])
    xyxy_boxes[:, 2] = np.clip(xyxy_boxes[:, 2], 0, img_shape_hw[1])
    xyxy_boxes[:, 3] = np.clip(xyxy_boxes[:, 3], 0, img_shape_hw[0])


class YoloInferece:
    """Class for encapsulating inference of YoloV7 model via ONNX"""

    def __init__(self, model_path: str, input_resolution: Tuple[int, int]):
        self.ort_sess = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_resolution = input_resolution

    def _preprocess_before_iference(self, image: np.ndarray) -> np.ndarray:
        letterboxed_image, _, _ = letterbox(image, new_shape=self.input_resolution)
        letterboxed_image = letterboxed_image.astype(np.float32)
        letterboxed_image /= 255.0
        letterboxed_image = np.transpose(letterboxed_image, (2, 0, 1))
        letterboxed_image = np.expand_dims(letterboxed_image, 0)
        return letterboxed_image

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Returns predictions in the format: [xyxy_box, conf]"""
        network_input = self._preprocess_before_iference(image)
        image_height, image_width = image.shape[:2]
        outputs = self.ort_sess.run(None, {"images": network_input})[0]

        boxes = outputs[:, 1:5]
        boxes = scale_boxes_to_original_image_space(
            input_img_hw_shape=self.input_resolution,
            xyxy_boxes=boxes,
            orig_image_hw_shape=(image_height, image_width),
        )
        confidences = outputs[:, -1]
        confidences = np.expand_dims(confidences, 1)
        predictions = np.concatenate((boxes, confidences), axis=1)
        return predictions

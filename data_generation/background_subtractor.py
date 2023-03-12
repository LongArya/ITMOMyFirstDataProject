import cv2
import os
from const import SCRIPT_DIR
from typing import List
import matplotlib.pyplot as plt
import numpy as np


# TODO refactor thiss
class BackgroundSubtractor:
    def __init__(self, calibration_image_path: str):
        self.calibration_image_path = calibration_image_path
        self.threshold: float = 0.0002
        self._superpixel_size_limit = 250
        self._desired_superpixel_num = 100
        self._superpixel_slic_iter_num = 15
        self._enforce_size_percents = 55

    @staticmethod
    def read_hsv_from_file(file: str) -> np.ndarray:
        """Reads image and converts it to opencv"""
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return image

    @staticmethod
    def read_rgb_from_file(file: str) -> np.ndarray:
        """Reads image and converts it to rgb"""
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _compute_backprojection_map_h_s_channels(
        self, roi_image_path: str, full_image_path: str
    ) -> np.ndarray:
        roi_hsv = BackgroundSubtractor.read_hsv_from_file(roi_image_path)
        full_image_hsv = BackgroundSubtractor.read_hsv_from_file(full_image_path)
        roihist = cv2.calcHist([roi_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        roi_h, roi_w = roi_hsv.shape[:2]
        roi_pixels_num = roi_h * roi_w
        backprpojection_map = cv2.calcBackProject(
            [full_image_hsv], [0, 1], roihist, [0, 180, 0, 256], scale=1
        )
        backprpojection_map = backprpojection_map.astype(np.float32)
        # convert counts to probabilities
        backprpojection_map /= roi_pixels_num
        return backprpojection_map

    def _get_superpixels_segmentation(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        image_LAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        image_h, image_w = image_LAB.shape[:2]
        image_area = image_h * image_w
        block_size_pixels = int(image_area / self._desired_superpixel_num)
        block_size_pixels = min(block_size_pixels, self._superpixel_size_limit)
        slic_obj = cv2.ximgproc.createSuperpixelSLIC(
            image=image_LAB,
            algorithm=cv2.ximgproc.SLIC,
            region_size=block_size_pixels,
        )
        slic_obj.iterate(self._superpixel_slic_iter_num)
        slic_obj.enforceLabelConnectivity(self._enforce_size_percents)
        labels_map = slic_obj.getLabels()
        contours_map = slic_obj.getLabelContourMask()
        plt.imshow(contours_map)
        plt.show()
        return labels_map

    def _get_superpixels_based_segmentation_mask(
        self,
        labels_map: np.ndarray,
        backprpojection_map: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        superpixels_num = np.max(labels_map)
        superpixels_mask = np.zeros(backprpojection_map.shape)
        for superpixel_ind in range(superpixels_num):
            y_indexes, x_indexes = np.nonzero(labels_map == superpixel_ind)
            probability_values = backprpojection_map[y_indexes, x_indexes]
            mean_superpixel_probability = probability_values.mean()
            if mean_superpixel_probability > threshold:
                superpixels_mask[y_indexes, x_indexes] = 1
        return superpixels_mask

    def solve(self, image_path: str, thrd: float) -> np.ndarray:
        skin_backprojection_map = self._compute_backprojection_map_h_s_channels(
            roi_image_path=self.calibration_image_path, full_image_path=image_path
        )
        skin_backprojection_map[skin_backprojection_map < thrd] = 0
        skin_backprojection_map[skin_backprojection_map > thrd] = 255
        return skin_backprojection_map

    def solve_with_superpixels(self, image_path: str, thrd: float) -> np.ndarray:
        superpixels_segmentation = self._get_superpixels_segmentation(image_path)
        skin_backprojection_map = self._compute_backprojection_map_h_s_channels(
            roi_image_path=self.calibration_image_path, full_image_path=image_path
        )
        superpixels_skin_mask = self._get_superpixels_based_segmentation_mask(
            labels_map=superpixels_segmentation,
            backprpojection_map=skin_backprojection_map,
            threshold=thrd,
        )
        return superpixels_skin_mask


def plot_histogram(histogram: List[float], axis: plt.Axes) -> None:
    """Plots histogram at given axis"""
    histogram_x_range: List[float] = list(range(len(histogram)))
    axis.bar(x=histogram_x_range, height=histogram)


def compute_hue_hist(image_path: str) -> np.ndarray:
    hsv_image = BackgroundSubtractor.read_hsv_from_file(image_path)
    hist = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])
    hist = np.squeeze(hist)
    return hist


def debug_histograms():
    background_image = os.path.join(SCRIPT_DIR, "background.jpg")
    background_image_v2 = os.path.join(SCRIPT_DIR, "background_v2.jpg")
    test_bg = background_image_v2
    hand_image = os.path.join(SCRIPT_DIR, "hand.jpg")

    fig, (bg_ax, hand_ax) = plt.subplots(1, 2)
    bg_hist = compute_hue_hist(test_bg)
    hand_hist = compute_hue_hist(hand_image)
    plot_histogram(histogram=bg_hist, axis=bg_ax)
    plot_histogram(histogram=hand_hist, axis=hand_ax)
    plt.show()


def test_solver():
    background_image = os.path.join(SCRIPT_DIR, "background.jpg")
    background_image_v2 = os.path.join(SCRIPT_DIR, "background_v2.jpg")
    test_bg = background_image_v2
    hand_image_path = os.path.join(SCRIPT_DIR, "hand.jpg")
    bg_subtractor = BackgroundSubtractor(calibration_image_path=test_bg)
    mask = bg_subtractor.solve_with_superpixels(hand_image_path, thrd=0.0)
    # mask = bg_subtractor.solve(hand_image, thrd=0.0001)
    # plt.imshow(mask)
    # plt.show()
    rgb_hand = BackgroundSubtractor.read_rgb_from_file(hand_image_path)
    rgb_hand[mask == 1] = [0, 0, 0]
    plt.imshow(rgb_hand)
    plt.show()


if __name__ == "__main__":
    test_solver()

import numpy as np
import os
import torch
from typing import List, Optional, Dict, Iterable, Tuple
from general.datasets.read_meta_dataset import ReadMetaDataset
from matplotlib.axes._axes import Axes
import torchvision.transforms as tf
import time


class TorchNormalizeInverse:
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        self.normalization_inverse = tf.Normalize(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return self.normalization_inverse(tensor.clone())


def timing_decorator(func):
    def wrapper(*arg, **kw):
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        print(f"{func.__name__} took {t2 - t1} seconds")
        return res

    return wrapper


def traverse_all_files(data_root: str) -> List[str]:
    all_files: List[str] = []
    for outer, inner, files in os.walk(data_root):
        for f in files:
            all_files.append(os.path.join(outer, f))
    return all_files


def keep_files_with_extension(files: List[str], extension: str) -> List[str]:
    kept_files: List[str] = list(
        filter(lambda p: os.path.splitext(p)[1] == extension, files)
    )
    return kept_files


def get_new_pattern_name_folder(root: str, pattern: str) -> str:
    """Get new folder with name like {pattern}_{num}"""
    if not os.path.exists(root):
        return os.path.join(root, f"{pattern}_{0:05d}")
    root_content = os.listdir(root)
    pattern_named_dirs = sorted(
        list(filter(lambda path: path.startswith(pattern), root_content))
    )
    if not pattern_named_dirs:
        return os.path.join(root, f"{pattern}_{0:05d}")
    max_pattern_name = os.path.splitext(pattern_named_dirs[-1])[0]
    max_pattern_num = int(max_pattern_name.split(f"_")[-1])
    return os.path.join(root, f"{pattern}_{max_pattern_num + 1:05d}")


def get_sample_with_image_path(
    dataset: ReadMetaDataset, image_path: str
) -> Optional[Dict]:
    for i in range(len(dataset)):
        cur_image_path = dataset.read_meta(i)["image_path"]
        if cur_image_path == image_path:
            return dataset[i]
    return None


def plot_images_in_grid(axes: List[List[Axes]], images: List[np.ndarray]) -> None:
    col_num = len(axes)
    row_num = len(axes[0])
    if len(images) > col_num * row_num:
        raise ValueError(
            f"Cannot draw {len(images)} images on {col_num}x{row_num} grid"
        )
    for image_index, image in enumerate(images):
        i, j = np.unravel_index(image_index, (col_num, row_num))
        axes[i][j].imshow(image)
    # turn off axis
    for i in range(col_num):
        for j in range(row_num):
            axes[i][j].axis("off")


def xyxy_box_area(xyxy_box: Iterable[float]) -> float:
    x1, y1, x2, y2 = xyxy_box
    widtdh = x2 - x1
    height = y2 - y1
    return widtdh * height


def get_intervals_intersection(
    interval1: Tuple[float, float], interval2: Tuple[float, float]
) -> float:
    start1, end1 = interval1
    start2, end2 = interval2
    iou = min(end2, end1) - max(start1, start2)
    iou = max(iou, 0)
    return iou


def get_xyxy_boxes_iou(xyxy_box1: Iterable[float], xyxy_box2: Iterable[float]) -> float:
    box1_x1, box1_y1, box1_x2, box1_y2 = xyxy_box1
    box2_x1, box2_y1, box2_x2, box2_y2 = xyxy_box2
    intersection_area = get_intervals_intersection(
        interval1=(box1_x1, box1_x2), interval2=(box2_x1, box2_x2)
    ) * get_intervals_intersection(
        interval1=(box1_y1, box1_y2), interval2=(box2_y1, box2_y2)
    )
    area1 = xyxy_box_area(xyxy_box1)
    area2 = xyxy_box_area(xyxy_box2)
    union_area = area1 + area2 - intersection_area
    iou = intersection_area / union_area
    return iou

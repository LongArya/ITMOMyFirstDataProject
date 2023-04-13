from PIL import Image
import torchvision.transforms as tf
import torch
import numpy as np
from const import IMAGENET_MEAN, IMAGENET_STD
from general.utils import TorchNormalizeInverse


def test_inverse_normalization():
    image = np.zeros((224, 224, 3))
    original_tensor = tf.ToTensor()(image)
    normalization = tf.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    inverse_normalization = TorchNormalizeInverse(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    normalized_tensor = normalization(original_tensor)
    denormalized_tensor = inverse_normalization(normalized_tensor)
    closeness_mask = torch.isclose(denormalized_tensor, original_tensor, atol=1e-05)
    assert torch.all(closeness_mask)

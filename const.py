import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(SCRIPT_DIR)
REPO_DIR = os.path.dirname(WORKSPACE_DIR)
DATA_ROOT = os.path.join(WORKSPACE_DIR, "Data")
NDRCZC_DATA_ROOT = os.path.join(DATA_ROOT, "ndrczc35bt-1")
CUSTOM_DATA_ROOT = os.path.join(DATA_ROOT, "custom_data")
CUSTOM_TRAIN_ROOT = os.path.join(CUSTOM_DATA_ROOT, "train")
CUSTOM_VAL_ROOT = os.path.join(CUSTOM_DATA_ROOT, "val")
CUSTOM_VAL_VIZ_ROOT = os.path.join(CUSTOM_DATA_ROOT, "val_viz")
CUSTOM_TRAIN_VIZ_ROOT = os.path.join(CUSTOM_DATA_ROOT, "train_viz")
CUSTOM_PRESPLIT_ROOT = os.path.join(CUSTOM_DATA_ROOT, "presplit_data")

HAGRID_DATA_ROOT = os.path.join(DATA_ROOT, "Hagrid_data")
HAGRID_IMAGES_ROOT = os.path.join(HAGRID_DATA_ROOT, "subsample")
HAGRID_META_ROOT = os.path.join(HAGRID_DATA_ROOT, "ann_subsample")

YOLO_V7_INPUT_RESOLUTION = (320, 320)
YOLO_V7_ONNX_HAND_DETECTION = os.path.join(
    SCRIPT_DIR, "MVP", "onnx_networks", "weights", "YoloV7_Tiny.onnx"
)
RESNET18_ONNX_CLASSIFIER = os.path.join(
    SCRIPT_DIR,
    "MVP",
    "onnx_networks",
    "weights",
    "checkpoint_epoch=49-val_weighted_F1=0.88.ckpt.onnx",
)


TRAIN_RESULTS_ROOT = os.path.join(WORKSPACE_DIR, "training_results")
STATIC_GESTURE_RESULTS_ROOT = os.path.join(
    TRAIN_RESULTS_ROOT, "static_gesture_classification"
)
STATIC_GESTURE_CFG_ROOT = os.path.join(
    WORKSPACE_DIR, "code", "static_gesture_classification", "conf"
)
STATIC_GESTURE_CFG_NAME = "base_static_gesture_config"
NEPTUNE_API_TOKEN_ENV = "NEPTUNE_API_TOKEN"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RESNET18_INPUT_SIZE = (224, 224)

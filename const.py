import os

SCRIPT_DIR = os.path.abspath(__file__)
WORKSPACE_DIR = os.path.dirname(SCRIPT_DIR)
REPO_DIR = os.path.dirname(WORKSPACE_DIR)
DATA_ROOT = os.path.join(REPO_DIR, "Data")

TRAIN_RESULTS_ROOT = os.path.join(REPO_DIR, "training_results")
STATIC_GESTURE_RESULTS_ROOT = os.path.join(
    TRAIN_RESULTS_ROOT, "static_gesture_classification"
)
STATIC_GESTURE_CFG_ROOT = os.path.join(
    WORKSPACE_DIR, "static_gesture_classification", "conf"
)
STATIC_GESTURE_CFG_NAME = "base_static_gesture_config"

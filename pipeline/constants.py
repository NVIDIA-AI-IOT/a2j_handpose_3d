DATASET = "NYU" # "Personal", "NYU"
DATSET_SWITCHER = {
    "Personal": {
        "dataset_path": "/home/analog/Desktop/NVIDIA/HandDepthData/data/depth_imgs",
        "joint_json_path": "/home/analog/Desktop/NVIDIA/HandDepthData/data/full_annotations",
    },

    "NYU": {
        "dataset_path": "/home/analog/Desktop/DataSets/NYU_dataset",
        "joint_json_path": "",
    }
}

DEPTH_IMG_PATH = DATSET_SWITCHER[DATASET]["dataset_path"]
JOINT_JSON_PATH = DATSET_SWITCHER[DATASET]["joint_json_path"]

# Set the checkpoint directory
SAVE_PATH = "/home/analog/Desktop/NVIDIA/hand_a2j/check_point"


# Select which backbone network you want to train with
BACKBONE_NAME_SWITCHER = {
    "resnet18": False,
    "resnet34": False,
    "resnet50": True,
    "resnet101": False,
    "resnet152": False,
    "mobilenet": False,
}

BACKBONE_NAME = [elem[0] for idx, elem in enumerate(BACKBONE_NAME_SWITCHER.items()) if elem[1]][0]
PRE_TRAINED = True # Set to true for pre trained backbone on ImageNet

DATA_SEGMENT = "1" # ALL, 1

NUM_JOINTS = 16 # 16, (36 Total NYU), 21 for personal
JOINT_LIST = [0, 2, 5, 6, 8, 11, 12, 14, 17, 18, 20, 23, 24, 26, 28, 34]
if NUM_JOINTS == 36:
    JOINT_LIST = [i for i in range(36)]
"""
pinky 0, 2, 5
ring 6, 8, 11
middle 12, 14, 17
index 18, 20, 23
thumb 24, 26, 28
palm 34
"""

TARGET_SIZE = (176, 176)
DEPTH_THRESHOLD = 180
RAND_CROP_SHIFT = 5
RANDOM_ROTATE = 180
RAND_SCALE = (1., 0.5)

TRAIN_VAL_SPLIT = 90
SAVE_FREQ = 1
MAX_EPOCH = 34

BATCH_SIZE = 64
IS_3D = True
STRIDE = 16

# LR_RATE = 0.0004
LR_RATE = 0.00035
WEIGHT_DECAY = 1e-4
STEP_SIZE = 10

GAMMA = 0.2
SPACIAL_FACTOR = 0.5
REG_LOSS_FACTOR = 3
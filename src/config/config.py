import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Data paths
DATASET_PATH = os.path.join(ROOT_DIR, "dataset")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")

# Model hyperparameters
IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 50
N_CLASSES = 3

# Split ratios
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
SHUFFLE_SIZE = 10000
SEED = 12

# Class names
CLASS_NAMES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]

# Hugging Face
HF_REPO_ID = "24f2004275/potato-disease-classifier"
HF_MODEL_FILENAME = "potato-disease-model.keras"

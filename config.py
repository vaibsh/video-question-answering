import os

class Config:
    # Base directory (can be overridden in Colab)
    BASE_DIR = os.environ.get("BASE_DIR", os.getcwd())

    # Data paths
    DATA_DIR = os.path.join(BASE_DIR, "data")
    VIDEO_DIR = os.path.join(DATA_DIR, "video")

    TRAIN_JSON = os.path.join(DATA_DIR, "train.json")
    VAL_JSON = os.path.join(DATA_DIR, "val.json")

    # Model paths
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    MODEL_PATH = os.path.join(MODEL_DIR, "model-video-q-a.pt")

    # Hyper-Params
    BATCH_SIZE = 32
    MAX_FRAMES = 16
    NUM_WORKERS = 4
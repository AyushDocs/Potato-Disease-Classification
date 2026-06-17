import os
import json
from datetime import datetime
from src.config.config import ARTIFACTS_DIR


def save_training_history(history, filename=None):
    if filename is None:
        filename = f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = os.path.join(ARTIFACTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(history.history, f, indent=2)
    return path


def save_metrics(scores, filename=None):
    if filename is None:
        filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = os.path.join(ARTIFACTS_DIR, filename)
    metrics = {"loss": float(scores[0]), "accuracy": float(scores[1])}
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path

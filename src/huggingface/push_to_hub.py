import os
import keras
from huggingface_hub import HfApi, create_repo, upload_file
from src.config.config import MODELS_DIR, HF_REPO_ID, HF_MODEL_FILENAME


def get_latest_keras_model():
    keras_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".keras")]
    if not keras_files:
        raise FileNotFoundError("No .keras model files found in models/ directory")
    versions = [(f, int(f.split(".")[0])) for f in keras_files if f.split(".")[0].isdigit()]
    versions.sort(key=lambda x: x[1])
    return os.path.join(MODELS_DIR, versions[-1][0])


def push_to_hub():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")

    model_path = get_latest_keras_model()
    model = keras.models.load_model(model_path)
    model.save(HF_MODEL_FILENAME)

    api = HfApi()
    create_repo(repo_id=HF_REPO_ID, token=hf_token, exist_ok=True, repo_type="model")

    upload_file(
        path_or_fileobj=HF_MODEL_FILENAME,
        path_in_repo=HF_MODEL_FILENAME,
        repo_id=HF_REPO_ID,
        token=hf_token,
    )

    readme_content = """---
license: mit
language: en
tags:
- potato-disease
- image-classification
- tensorflow
- cnn
- keras
datasets:
- PlantVillage
metrics:
- accuracy
---

# Potato Disease Classifier

CNN model for classifying potato leaf diseases: Early Blight, Late Blight, and Healthy.

## Model Description

A Sequential CNN built with TensorFlow/Keras that classifies potato leaf images into three categories. The model includes built-in preprocessing (resizing + rescaling) and data augmentation layers.

## Architecture

- Input preprocessing: Resizing(256,256) + Rescaling(1/255)
- Data augmentation: RandomFlip + RandomRotation (training only)
- 6x Conv2D(32-64 filters, 3x3, ReLU) + MaxPooling2D(2x2) blocks
- Flatten + Dense(64, ReLU) + Dense(3, Softmax)
- Total params: 183,747

## Classes

| Class | Description |
|-------|-------------|
| `Potato___Early_blight` | Early blight disease infection |
| `Potato___Late_blight`  | Late blight disease infection |
| `Potato___healthy`      | Healthy potato leaf |

## Performance

- Test Accuracy: ~99.6%
- Test Loss: ~0.009
- Validation Accuracy: ~99.5%
- Validation Loss: ~0.011

## Dataset

Trained on the [Potato Dataset](https://www.kaggle.com/datasets/faysalmiah1721758/potato-dataset) from Kaggle (PlantVillage subset), containing 2,152 images across 3 classes.

## Usage

```python
import keras
import numpy as np

model = keras.models.load_model("potato-disease-model.keras")

def predict(image_path):
    img = keras.utils.load_img(image_path, target_size=(256, 256))
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)
    class_names = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * float(np.max(predictions[0])), 2)
    return predicted_class, confidence
```
"""

    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=HF_REPO_ID,
        token=hf_token,
    )

    os.remove(HF_MODEL_FILENAME)
    print(f"Model pushed to https://huggingface.co/{HF_REPO_ID}")


if __name__ == "__main__":
    push_to_hub()

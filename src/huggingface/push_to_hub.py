import os
import keras
import tensorflow as tf
import numpy as np
from huggingface_hub import HfApi, create_repo, upload_file
from src.config.config import MODELS_DIR, HF_REPO_ID, HF_MODEL_FILENAME, IMAGE_SIZE


def get_latest_model_path():
    versions = [int(d) for d in os.listdir(MODELS_DIR) if d.isdigit()]
    if not versions:
        raise FileNotFoundError("No saved models found in models/ directory")
    latest = max(versions)
    return os.path.join(MODELS_DIR, str(latest))


def convert_savedmodel_to_keras(model_path):
    imported = tf.saved_model.load(model_path)
    model = tf.keras.Sequential([
        keras.layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        keras.layers.Rescaling(1.0 / 255),
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomRotation(0.2),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(3, activation='softmax'),
    ])
    model.build((None, IMAGE_SIZE, IMAGE_SIZE, 3))

    saved_weights = []
    for i in range(8):
        lw = getattr(imported, f"layer_with_weights-{i}")
        saved_weights.append(lw.variables[0].numpy())
        saved_weights.append(lw.variables[1].numpy())

    trainable_layers = [l for l in model.layers if len(l.get_weights()) > 0]
    for i, l in enumerate(trainable_layers):
        l.set_weights([saved_weights[i * 2], saved_weights[i * 2 + 1]])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'],
    )
    return model


def push_to_hub():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")

    model_path = get_latest_model_path()
    model = convert_savedmodel_to_keras(model_path)
    model.save(HF_MODEL_FILENAME)
    if os.path.exists("potato-disease-model.h5"):
        os.remove("potato-disease-model.h5")

    api = HfApi()
    create_repo(repo_id=HF_REPO_ID, token=hf_token, exist_ok=True, repo_type="model")

    upload_file(
        path_or_fileobj=HF_MODEL_FILENAME,
        path_in_repo=HF_MODEL_FILENAME,
        repo_id=HF_REPO_ID,
        token=hf_token,
    )

    readme_content = f"""---
license: mit
language: en
tags:
- potato-disease
- image-classification
- tensorflow
- cnn
datasets:
- PlantVillage
metrics:
- accuracy
---

# Potato Disease Classifier

CNN model for classifying potato leaf diseases: Early Blight, Late Blight, and Healthy.

## Model Description

A Sequential CNN built with TensorFlow/Keras that classifies potato leaf images into three categories.

## Classes

- `Potato___Early_blight`
- `Potato___Late_blight`
- `Potato___healthy`

## Performance

- Test Accuracy: ~98.8%
- Test Loss: ~0.038

## Usage

```python
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("potato-disease-model.h5")

def predict(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
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

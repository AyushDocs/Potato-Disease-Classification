# Potato Disease Classification

A CNN to classify potato leaf images into Early Blight, Late Blight, or Healthy categories using TensorFlow/Keras.

**Live Demo:** [Potato Disease Classifier](https://potato-disease-classifier.streamlit.app/)

## Model Architecture

Sequential CNN with built-in preprocessing and data augmentation:

- Input preprocessing: Resizing to 256×256 + Rescaling to [0,1]
- Data augmentation: RandomFlip + RandomRotation (training only)
- 6× Conv2D (32–64 filters, 3×3, ReLU) + MaxPooling2D (2×2)
- Flatten → Dense(64, ReLU) → Dense(3, Softmax)
- **Total params:** 183,747

**Performance:** ~99.6% test accuracy, ~0.009 test loss.

## Classes

| Class | Description |
|-------|-------------|
| `Potato___Early_blight` | Early blight disease infection |
| `Potato___Late_blight` | Late blight disease infection |
| `Potato___healthy` | Healthy potato leaf |

## Dataset

Trained on the [Potato Dataset](https://www.kaggle.com/datasets/faysalmiah1721758/potato-dataset) from Kaggle (PlantVillage subset) — 2,152 images across 3 classes.

## Project Structure

```
├── src/
│   ├── pipeline/
│   │   ├── data_ingestion.py      # Load & split dataset
│   │   ├── data_preprocessing.py   # Resize, rescale, augment
│   │   └── model_training.py       # Build, compile, train, save
│   ├── config/
│   │   └── config.py               # Hyperparameters & paths
│   ├── inference/
│   │   └── predictor.py            # Load model & predict
│   ├── huggingface/
│   │   └── push_to_hub.py          # Push model to Hugging Face
│   └── utils/
│       └── helpers.py              # Utility functions
├── notebooks/
│   └── 01-AyushDocs-ModelTraining.ipynb  # Training notebook
├── artifacts/                      # Training artifacts
├── models/                         # Saved models (.keras format)
├── dataset/                        # Training data
├── app.py                          # Streamlit web app
├── requirements.txt
└── setup.py
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

Run the full pipeline:
```bash
python -m src.pipeline.data_ingestion
python -m src.pipeline.model_training
```

Or train interactively in the [notebook](notebooks/01-AyushDocs-ModelTraining.ipynb).

## Inference

```python
from src.inference.predictor import PotatoDiseasePredictor

predictor = PotatoDiseasePredictor("models/1.keras")
class_name, confidence = predictor.predict("path/to/image.jpg")
```

## Streamlit App

```bash
streamlit run app.py
```

The app loads the model from Hugging Face Hub — upload a potato leaf image and get instant classification with confidence scores.

## Push to Hugging Face Hub

```bash
export HF_TOKEN="your_huggingface_token"
python -m src.huggingface.push_to_hub
```

# Potato Disease Classification

A deep learning-based system that classifies potato leaf images into **Early Blight**, **Late Blight**, or **Healthy** categories using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

## Model Architecture

The model is a Sequential CNN with 6 Conv2D + MaxPooling2D blocks followed by Dense layers:

- Image resizing & rescaling (256×256, pixel values normalized to [0,1])
- Data augmentation (random flip, rotation)
- 6 convolutional layers with ReLU activation and max pooling
- Flatten + Dense(64) + Dense(3) with softmax output

**Performance**: ~98.8% accuracy on test set.

## Classes

| Class | Description |
|-------|-------------|
| `Potato___Early_blight` | Early blight disease infection |
| `Potato___Late_blight` | Late blight disease infection |
| `Potato___healthy` | Healthy potato leaf |

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
├── artifacts/                      # Training artifacts
├── models/                         # Saved models (versioned)
├── dataset/                        # Training data
├── requirements.txt
├── setup.py
└── model.ipynb                     # Original training notebook
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

Or run inference on a single image:
```python
from src.inference.predictor import PotatoDiseasePredictor

predictor = PotatoDiseasePredictor("models/1")
class_name, confidence = predictor.predict("path/to/image.jpg")
```

## Streamlit App

Run the web UI:
```bash
streamlit run app.py
```

The app loads the model directly from Hugging Face Hub — just upload a potato leaf image and get instant classification with confidence scores.

## Push to Hugging Face

Set your token and run:
```bash
export HF_TOKEN="your_token"
python -m src.huggingface.push_to_hub
```

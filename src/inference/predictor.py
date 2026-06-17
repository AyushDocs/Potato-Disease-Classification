import numpy as np
import tensorflow as tf
from src.config.config import IMAGE_SIZE, CLASS_NAMES


class PotatoDiseasePredictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = CLASS_NAMES

    def predict(self, image_path):
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class = self.class_names[np.argmax(predictions[0])]
        confidence = round(100 * float(np.max(predictions[0])), 2)
        return predicted_class, confidence

    def predict_batch(self, image_paths):
        results = []
        for path in image_paths:
            cls, conf = self.predict(path)
            results.append({"image": path, "class": cls, "confidence": conf})
        return results

import os
import tensorflow as tf
from src.config.config import IMAGE_SIZE, BATCH_SIZE, CHANNELS, EPOCHS, N_CLASSES, MODELS_DIR
from src.pipeline.data_ingestion import get_partitioned_datasets
from src.pipeline.data_preprocessing import build_resize_and_rescale, build_data_augmentation, prepare_datasets
from src.utils.helpers import save_training_history, save_metrics


def build_model():
    input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    model = tf.keras.Sequential([
        build_resize_and_rescale(),
        build_data_augmentation(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(N_CLASSES, activation='softmax'),
    ])
    model.build(input_shape=input_shape)
    return model


def train():
    train_ds, val_ds, test_ds, class_names = get_partitioned_datasets()
    train_ds, val_ds, test_ds = prepare_datasets(train_ds, val_ds, test_ds)

    model = build_model()
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'],
    )

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        validation_data=val_ds,
    )

    scores = model.evaluate(test_ds)
    print(f"Test loss: {scores[0]}, Test accuracy: {scores[1]}")

    save_training_history(history)
    save_metrics(scores)

    model_version = max([int(i) for i in os.listdir(MODELS_DIR) + ["0"]]) + 1
    save_path = os.path.join(MODELS_DIR, str(model_version))
    model.save(save_path)
    print(f"Model saved to {save_path}")

    return model, history, scores


if __name__ == "__main__":
    train()

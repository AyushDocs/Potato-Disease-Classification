import tensorflow as tf
from src.config.config import IMAGE_SIZE


def build_resize_and_rescale():
    return tf.keras.Sequential([
        tf.keras.layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        tf.keras.layers.Rescaling(1.0 / 255),
    ])


def build_data_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
    ])


def prepare_datasets(train_ds, val_ds, test_ds):
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, test_ds

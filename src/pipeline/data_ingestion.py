import os
import tensorflow as tf
from src.config.config import DATASET_PATH, IMAGE_SIZE, BATCH_SIZE


def load_dataset():
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_PATH,
        shuffle=True,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
    )
    return dataset


def split_dataset(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000, seed=12):
    ds_size = len(ds)
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=seed)
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    return train_ds, val_ds, test_ds


def get_partitioned_datasets():
    dataset = load_dataset()
    train_ds, val_ds, test_ds = split_dataset(dataset)
    return train_ds, val_ds, test_ds, dataset.class_names


if __name__ == "__main__":
    train_ds, val_ds, test_ds, class_names = get_partitioned_datasets()
    print(f"Classes: {class_names}")
    print(f"Train batches: {len(train_ds)}")
    print(f"Val batches: {len(val_ds)}")
    print(f"Test batches: {len(test_ds)}")

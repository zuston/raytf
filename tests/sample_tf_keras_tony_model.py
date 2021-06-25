"""
From TonY example: https://github.com/linkedin/TonY/blob/master/tony-examples/mnist-tensorflow/mnist_keras_distributed.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import numpy as np
import tensorflow as tf


def mnist_dataset(batch_size):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    # The `x` arrays are in uint8 and have values in the range [0, 255].
    # You need to convert them to float32 with values in the range [0, 1]
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
    return train_dataset


def build_and_compile_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28)),
        tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10)
    ])
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=0.05,
            momentum=0.5),
        metrics=["accuracy"])
    return model


def get_num_workers():
    """Retrieve the number of workers in the training job."""
    if "TF_CONFIG" not in os.environ:
        return 1
    tf_config = json.loads(os.environ["TF_CONFIG"])
    num_workers = len(tf_config["cluster"]["worker"])
    return num_workers


def train_mnist():
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    per_worker_batch_size = 64
    num_workers = get_num_workers()
    global_batch_size = per_worker_batch_size * num_workers
    multi_worker_dataset = mnist_dataset(global_batch_size)
    with strategy.scope():
        multi_worker_model = build_and_compile_cnn_model()

    multi_worker_model.fit(
        multi_worker_dataset,
        epochs=2,
        steps_per_epoch=70,
    )


def tony_keras_process(args):
    train_mnist()


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
    train_mnist()

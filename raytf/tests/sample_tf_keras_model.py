import os
import json
import numpy as np
import tensorflow as tf

FLAGS = tf.compat.v1.app.flags.FLAGS

# 路径相关
tf.compat.v1.app.flags.DEFINE_string("data_dir", '', "data dir")
tf.compat.v1.app.flags.DEFINE_string("output_dir", '', "output dir")
tf.compat.v1.app.flags.DEFINE_string("model_dir", '', "model dir")


def mnist_dataset(batch_size):
    '''
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    # The `x` arrays are in uint8 and have values in the range [0, 255].
    # You need to convert them to float32 with values in the range [0, 1]
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
    return train_dataset
    '''
    features = tf.constant([[1.0, 3.0], [2.0, 1.0], [3.0, 3.0]])
    labels = tf.constant([4.0, 3.0, 6.0])
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (features, labels)).repeat(100000).batch(batch_size)
    return train_dataset


def build_and_compile_cnn_model():
    '''
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
    '''
    model = tf.keras.Sequential([
        tf.keras.Input(shape=2),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model


def get_num_workers():
    """Retrieve the number of workers in the training job."""
    if "TF_CONFIG" not in os.environ:
        return 1
    tf_config = json.loads(os.environ["TF_CONFIG"])
    num_workers = len(tf_config["cluster"]["worker"])
    return num_workers


def main(_):
    # data_dir = FLAGS.data_dir
    # model_dir = FLAGS.model_dir
    model_dir = "/tmp/opal/keras/1"
    print('model_dir is ' + str(model_dir))
    # output_dir = FLAGS.output_dir

    per_worker_batch_size = 64
    num_workers = get_num_workers()
    print('num_workers is ' + str(num_workers))
    global_batch_size = per_worker_batch_size * num_workers
    multi_worker_dataset = mnist_dataset(global_batch_size)
    strategy = tf.distribute.experimental.ParameterServerStrategy(
        cluster_resolver=tf.distribute.cluster_resolver.TFConfigClusterResolver())

    with strategy.scope():
        multi_worker_model = build_and_compile_cnn_model()

    multi_worker_model.fit(
        multi_worker_dataset,
        epochs=100,
        steps_per_epoch=70,
    )


def keras_process(args):
    main(None)


if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.app.run()

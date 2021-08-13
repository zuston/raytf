import numpy as np
import tensorflow as tf
from raytf.raytf_driver import Driver
import ray

class FLAGS:
    model_dir = "/tmp/opal/12"
    dropout = 0.5
    learning_rate = 0.001
    train_epoch = 2000
    batch_size = 64
    max_steps = 300


def build_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.int)
    y_train = y_train.astype(np.int)
    x_test = x_test.astype(np.int)
    y_test = y_test.astype(np.int)
    return x_train, y_train, x_test, y_test


def build_config():
    return tf.estimator.RunConfig(
        keep_checkpoint_max=3,
        save_checkpoints_steps=1000,
        save_summary_steps=500,
        log_step_count_steps=500
    )


def build_feature_columns():
    return [tf.feature_column.numeric_column("feature", shape=[28, 28])]


def build_classifier(feature_columns, run_config):
    return tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[256, 128],
        optimizer=tf.keras.optimizers.Adagrad(learning_rate=FLAGS.learning_rate, epsilon=1e-5),
        n_classes=10,
        dropout=FLAGS.dropout,
        model_dir=FLAGS.model_dir,
        config=run_config
    )


def build_input_fn(x_train, y_train, x_test, y_test):
    train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        {"feature": x_train},
        y_train,
        batch_size=FLAGS.batch_size,
        shuffle=True, num_epochs=FLAGS.train_epoch)

    eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        {"feature": x_test},
        y_test,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        num_epochs=1)

    return train_input_fn, eval_input_fn


def build_spec(train_input_fn, eval_input_fn):
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=FLAGS.max_steps,
        hooks=[])

    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        steps=500,
        throttle_secs=60)

    return train_spec, eval_spec


def main_action(argv):
    del argv
    tf.get_logger().setLevel('INFO')

    run_config = build_config()
    x_train, y_train, x_test, y_test = build_mnist()
    feature_columns = build_feature_columns()
    classifier = build_classifier(feature_columns, run_config)
    train_input_fn, eval_input_fn = build_input_fn(x_train, y_train, x_test, y_test)
    train_spec, eval_spec = build_spec(train_input_fn, eval_input_fn)

    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
    if run_config.task_type == "chief":
        classifier.evaluate(eval_input_fn)


if __name__ == '__main__':
    ray.init()
    tf_cluster = Driver.build(resources=
    {
        "ps": {"cores": 1, "memory": 1, "gpu": 2, "instances": 1},
        "worker": {"cores": 1, "memory": 1, "gpu": 2, "instances": 1},
        "chief": {"cores": 1, "memory": 1, "gpu": 2, "instances": 1}
    }
    )
    tf_cluster.start(model_process=main_action, args=None)
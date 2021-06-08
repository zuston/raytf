import sys
import typing
from typing import Dict

import logging
import ray
from typing import Tuple, List
import json
import os


@ray.remote(max_retries=0)
class Executor:
    def __init__(self, role_name: str, role_index: int):
        self.__logger = get_logger(__name__)
        self.__role_name = role_name
        self.__role_index = role_index
        # 初始化节点地址和 grpc 端口，用来构建 tf cluster spec.
        self.__node_ip = get_node_address()
        self.__grpc_port = get_reserved_grpc_port()

    def get_role_info(self) -> Tuple[str, int, str]:
        return self.__role_name, self.__role_index, f"{self.__node_ip}:{self.__grpc_port}"

    def set_tf_cluster_spec(self, spec_info: List[Tuple[str, int, str]]):
        spec_tmp_dict = {}
        for (role, _, hostip) in spec_info:
            if role not in spec_tmp_dict:
                spec_tmp_dict[role] = [hostip]
            else:
                spec_tmp_dict[role].append(hostip)
        tf_spec_dict = {
            "cluster": spec_tmp_dict,
            "task": {
                "type": self.__role_name,
                "index": self.__role_index
            }
        }
        cluster_spec_json = json.dumps(tf_spec_dict)
        self.__logger.info(f"[{self.__role_name}][{self.__role_index}], cluster spec: {cluster_spec_json}")
        os.environ['TF_GRPC_REUSE_PORT'] = "true"
        os.environ['TF_CONFIG'] = cluster_spec_json

    def run(self, model_process, args) -> Tuple[str, int]:
        self.__logger.info(f"[{self.__role_name}][{self.__role_index}], Running")
        # if self.__role_name == 'ps' and self.__role_index == 0:
        #     time.sleep(random.randint(10, 20))
        model_process(args)
        self.__logger.info(f"[{self.__role_name}][{self.__role_index}], Finished")
        return self.__role_name, self.__role_index


class TensorflowCluster:
    @staticmethod
    def build(resources: Dict[str, Dict[str, str]] = None):
        if not ray.is_initialized():
            ray.init(address='auto')
        tf_cluster = TensorflowCluster()
        tf_cluster.__build(resources=resources)
        return tf_cluster

    def __init__(self):
        self.__logger = get_logger(__name__)
        self.__logger.info("Init tensorflow cluster.")
        self.__role_executors_list = []
        self.__ps_size = 0

    '''
    resources will be as follows: (memory-unit GB)
        ps: {"cores": "2", "memory": "2", "gpu": "2", "instances": "2"}
        worker: {"cores": "2", "memory": "2", "gpu": "2", "instances": "4"}
        evaluator: {"cores": "2", "memory": "2", "gpu": "2", "instances": "1"}
    '''

    def __build(self, resources: Dict[str, Dict[str, str]] = None):
        if not resources:
            raise Exception("Must set the tensorflow cluster resources.")
        self.__logger.info(f"tf cluster resources: {resources}")
        self.__build_executor(resources)
        self.__logger.info("Startup tensorflow cluster.")

    def start(self, model_process, args):
        self.__logger.info("Starting training.")

        finished_no_ps_role_size = 0
        tracked_role_size = len(self.__role_executors_list) - self.__ps_size

        waiting = [executor.run.remote(model_process, args) for executor in self.__role_executors_list]
        stop = False
        while len(waiting) > 0 and not stop:
            ready, waiting = ray.wait(waiting, num_returns=1)
            finished_task_list = ray.get(ready)
            for role_name, _ in finished_task_list:
                if role_name != 'ps':
                    finished_no_ps_role_size += 1
            if finished_no_ps_role_size == tracked_role_size:
                self.__logger.info(f"All {finished_no_ps_role_size} tracked tasks have finished. Stop all PS...")
                stop = True

    def __build_executor(self, resources):
        for role_name, role_resources_dict in resources.items():
            # todo: 临时设置为默认资源
            cores = int(role_resources_dict["cores"]) if role_resources_dict["cores"] else 1
            memory = int(role_resources_dict["memory"]) if role_resources_dict["memory"] else 1
            memory_bytes = memory * 1024 * 1024 * 1024
            instances = int(role_resources_dict["instances"]) if role_resources_dict["instances"] else 1

            if role_name == 'ps':
                self.__ps_size = instances

            # executor_objs = [Executor.options(num_cpus=cores, memory=memory_bytes).remote(role_name, index)
            #                  for index in range(instances)]
            executor_objs = [Executor.remote(role_name, index) for index in range(instances)]
            self.__logger.info(f"Request resources. role_name: {role_name}, instances: {len(executor_objs)}")

            self.__role_executors_list.extend(executor_objs)

        role_info_list = ray.get([executor.get_role_info.remote() for executor in self.__role_executors_list])
        ray.get([executor.set_tf_cluster_spec.remote(role_info_list) for executor in self.__role_executors_list])


def get_logger(name, level="INFO", handlers=None, update=False):
    _DEFAULT_LOGGER = "ps.logger"

    _DEFAULT_FORMATTER = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] "
        "[%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
    )

    _ch = logging.StreamHandler(stream=sys.stdout)
    _ch.setFormatter(_DEFAULT_FORMATTER)

    _DEFAULT_HANDLERS = [_ch]

    _LOGGER_CACHE = {}  # type: typing.Dict[str, logging.Logger]

    if name in _LOGGER_CACHE and not update:
        return _LOGGER_CACHE[name]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = handlers or _DEFAULT_HANDLERS
    logger.propagate = False
    return logger


import psutil
import socket


def get_node_address() -> str:
    """
    Get the ip address used in ray.
    """
    pids = psutil.pids()
    for pid in pids:
        try:
            proc = psutil.Process(pid)
            for arglist in proc.cmdline():
                for arg in arglist.split(" "):
                    if arg.startswith("--node-ip-address"):
                        addr = arg.split("=")[1]
                        return addr
        except psutil.AccessDenied:
            pass
        except psutil.NoSuchProcess:
            pass
    raise Exception("can't find any ray process")


'''
设置为 TensorFlow 复用端口方案，防止在运行过程中被抢占
'''


def get_reserved_grpc_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    s.bind(('', 0))
    _, port = s.getsockname()
    return port


# flags.DEFINE_string('model_dir', 'output', 'path of model output')
# flags.DEFINE_float('dropout', 0.5, 'dropout of this model')
# flags.DEFINE_float('learning_rate', 0.001, 'learning rate of this model')
# flags.DEFINE_integer('train_epoch', 500, 'train epoch of this model')
# flags.DEFINE_integer('batch_size', 64, 'batch size of this model')
# flags.DEFINE_integer('max_steps', 300000, 'batch size of this model')
# FLAGS = flags.FLAGS

class FLAGS:
    model_dir = "/tmp/opal/4"
    dropout = 0.5
    learning_rate = 0.001
    train_epoch = 20
    batch_size = 64
    max_steps = 3000


def process_internal(args):
    import numpy as np
    import tensorflow as tf
    from absl import flags, app, logging

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

    del args
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

@ray.remote
def process_external():
    import model
    return model.process


if __name__ == '__main__':
    tf_cluster = TensorflowCluster.build(resources=
    {
        "ps": {"cores": "2", "memory": "2", "gpu": "2", "instances": "1"},
        "worker": {"cores": "2", "memory": "2", "gpu": "2", "instances": "1"},
        "chief": {"cores": "2", "memory": "2", "gpu": "2", "instances": "1"}
    }
    )
    tf_cluster.start(model_process=process_external.remote(), args=None)

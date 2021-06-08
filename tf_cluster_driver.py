import ray
from typing import Dict

from tf_executor import Executor
from log_utils import get_logger


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

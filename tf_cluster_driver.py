import ray
from typing import Dict, Tuple
import os
import tf_executor
import log_utils

UNTRACKED_ROLE_NAMES = {"PS": "", tf_executor.SIDECAR_TB_ROLE_NAME: ""}


class TensorflowCluster:
    @staticmethod
    def build(resources: Dict[str, Dict[str, str]] = None, event_log: str = None):
        if not ray.is_initialized():
            current_main_path = os.path.dirname(os.path.abspath(__file__))
            print(f"Current execution file path: [{current_main_path}], its py files "
                  f"will be distributed to other workers.")
            runtime_env = {
                "working_dir": current_main_path
            }
            jobconf = ray.job_config.JobConfig(runtime_env=runtime_env)
            # It will be attached to existed Ray cluster.
            ray.init(address='auto', job_config=jobconf)
        tf_cluster = TensorflowCluster()
        tf_cluster.__build(resources=resources, event_log_path=event_log)
        return tf_cluster

    def __init__(self):
        self.__logger = log_utils.get_logger(__name__)
        self.__logger.info("Init tensorflow cluster.")
        self.__role_executors_list = []
        self.__ps_size = 0
        self.__tb_url = None

    '''
    resources will be as follows: (memory-unit GB)
        ps: {"cores": "2", "memory": "2", "gpu": "2", "instances": "2"}
        worker: {"cores": "2", "memory": "2", "gpu": "2", "instances": "4"}
        evaluator: {"cores": "2", "memory": "2", "gpu": "2", "instances": "1"}
    '''

    def __build(self, resources: Dict[str, Dict[str, str]] = None, event_log_path: str = None):
        if not resources:
            raise Exception("Must set the tensorflow cluster resources.")
        self.__logger.info(f"tf cluster resources: {resources}")

        # When enabled event_log_path, it should request new resource for sidecar-tb
        self.__event_log_path = event_log_path
        self.__sidecar_tb_enabled = False
        if self.__event_log_path:
            self.__sidecar_tb_enabled = True
            resources[tf_executor.SIDECAR_TB_ROLE_NAME] = {"cores": "2", "memory": "2", "gpu": "0", "instances": "1"}
            self.__sidecar_tb_enabled = True

        self.__build_executor(resources)
        self.__logger.info("Startup tensorflow cluster.")

    def start(self, model_process, args):
        self.__logger.info("Starting training.")

        finished_no_ps_role_size = 0
        tracked_role_size = len(self.__role_executors_list) - self.__ps_size - (
            0 if not self.__sidecar_tb_enabled else 1)

        if self.__sidecar_tb_enabled and self.__tb_url:
            self.__logger.info(f"Sidecar tensorboard visiting url: {self.__tb_url}")

        waiting = [executor.run.remote(model_process, args) for executor in self.__role_executors_list]
        stop = False
        while len(waiting) > 0 and not stop:
            ready, waiting = ray.wait(waiting, num_returns=1)
            finished_task_list = ray.get(ready)
            for role_name, _ in finished_task_list:
                if role_name != 'ps' and role_name != tf_executor.SIDECAR_TB_ROLE_NAME:
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

            executor_objs = [tf_executor.Executor.options(name=f"{role_name}-{index}",
                                                          num_cpus=cores, memory=memory_bytes)
                                 .remote(role_name, index, self.__event_log_path)
                             for index in range(instances)]
            self.__logger.info(f"Request resources. role_name: {role_name}, instances: {len(executor_objs)}")

            self.__role_executors_list.extend(executor_objs)

        role_info_list = ray.get([executor.get_role_info.remote() for executor in self.__role_executors_list])
        self.__logger.info(f"TF cluster role info list: {role_info_list}")

        tb_urls = [tb_url for role_name, _, _, tb_url in role_info_list if
                   role_name == tf_executor.SIDECAR_TB_ROLE_NAME]
        if tb_urls and len(tb_urls) == 1:
            self.__tb_url = tb_urls[0]

        ray.get([executor.set_tf_cluster_spec.remote(role_info_list) for executor in self.__role_executors_list])

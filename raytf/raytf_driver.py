import ray
from typing import Dict, Tuple
from raytf import raytf_executor
from raytf import log_utils
from ray.util.placement_group import (
    placement_group,
    placement_group_table
)
import sys
import os

from raytf.tf_runtime import TFRuntime

UNTRACKED_ROLE_NAMES = {"PS": "", raytf_executor.SIDECAR_TB_ROLE_NAME: ""}

RAY_RUNTIME_WORK_DIR = "working_dir"

DL_FRAMEWORK = "framework"
DEFAULT_DL_FRAMEWORK = "tensorflow"


class Driver:
    @staticmethod
    def get_main_execution_path():
        try:
            main_execution_path = os.path.dirname(os.path.abspath(sys.modules['__main__'].__file__))
            print(f"Execution path [{main_execution_path}] py files will be distributed to Ray cluster.")
            return main_execution_path
        except KeyError:
            print('library not loaded from script')
        except AttributeError:
            print('script not loaded from file')
        return None

    @staticmethod
    def build(resources: Dict[str, Dict[str, str]] = None,
              event_log: str = None,
              resources_reserved_timeout: int = None):
        if not ray.is_initialized():
            # todo: Dont support in python interactive model
            main_execute_path = Driver.get_main_execution_path()
            runtime_env = {
                RAY_RUNTIME_WORK_DIR: main_execute_path
            }
            jobconf = ray.job_config.JobConfig(runtime_env=runtime_env)
            ray.init(address='auto', job_config=jobconf)

        cluster = Driver()
        cluster.__build(resources=resources,
                        event_log_path=event_log,
                        resources_reserved_timeout=resources_reserved_timeout)
        return cluster

    def __init__(self):
        self.__logger = log_utils.get_logger(__name__)
        self.__logger.info("Init raytf cluster.")
        self.__role_executors_list = []
        self.__ps_size = 0
        self.__tb_url = None
        self.__placement_group = None
        self.__resources_reserved_timeout = None

        self.__framework_runtime = TFRuntime()

    '''
    resources will be as follows: (memory-unit GB)
        ps: {"cores": 2, "memory": 2, "gpu": 2, "instances": 4}
        worker: {"cores": 2, "memory": 2, "gpu": 2, "instances": 2}
        evaluator: {"cores": 2, "memory": 2, "gpu": 2, "instances": 1}
    '''

    def __build(self,
                resources: Dict[str, Dict[str, str]] = None,
                event_log_path: str = None,
                resources_reserved_timeout: int = None):
        if not resources:
            raise Exception("Must set the raytf cluster resources.")

        self.__logger.info(f"Raytf cluster resources: {resources}")

        # When enabled event_log_path, it should request new resource for sidecar-tb
        self.__event_log_path = event_log_path
        self.__resources_reserved_timeout = resources_reserved_timeout

        self.__sidecar_tb_enabled = False
        if self.__event_log_path:
            self.__sidecar_tb_enabled = True
            resources[raytf_executor.SIDECAR_TB_ROLE_NAME] = {"cores": "2", "memory": "2", "gpu": "0", "instances": "1"}
            self.__sidecar_tb_enabled = True

        self.__reserved_resources(resources)
        self.__build_executor(resources)
        self.__logger.info("Startup raytf training cluster.")

    def start(self, model_process, args):
        self.__logger.info("Starting training.")

        if self.__sidecar_tb_enabled and self.__tb_url:
            self.__logger.info(f"Sidecar tensorboard visiting url: {self.__tb_url}")

        tracked_role_finished_size = 0
        need_tracked_role_list = self.__framework_runtime.get_tracked_role_list()
        tracked_role_size = self.get_tracked_role_size(self.__role_executors_list, need_tracked_role_list)

        self.__logger.info(f"Need tracked role name set: {need_tracked_role_list}, role size: {tracked_role_size}")

        stop = False

        waiting = [executor.run.remote(model_process, args) for executor in self.__role_executors_list]
        while len(waiting) > 0 and not stop:
            ready, waiting = ray.wait(waiting, num_returns=1)
            finished_task_list = ray.get(ready)
            for role_name, _ in finished_task_list:
                if role_name in need_tracked_role_list:
                    tracked_role_finished_size += 1
            if tracked_role_finished_size == tracked_role_size:
                self.__logger.info(f"All {tracked_role_finished_size} tracked tasks have finished. "
                                   f"Stop all other role, like PS...")
                stop = True

    def __build_executor(self, resources):
        for role_name, role_resources_dict in resources.items():
            cores, memory_bytes, instances = self.__parse_resources_conf(role_resources_dict)
            if role_name == 'ps':
                self.__ps_size = instances

            executor_objs = [raytf_executor
                                 .Executor
                                 .options(name=f"{role_name}-{index}",
                                          num_cpus=cores,
                                          memory=memory_bytes,
                                          placement_group=self.__placement_group
                                          )
                                 .remote(role_name,
                                         index, self.__event_log_path, runtime=self.__framework_runtime)
                             for index in range(instances)]
            self.__logger.info(f"Request resources. role_name: {role_name}, instances: {len(executor_objs)}")

            self.__role_executors_list.extend(executor_objs)

        role_info_list = ray.get([executor.get_role_info.remote() for executor in self.__role_executors_list])
        self.__logger.info(f"TF cluster role info list: {role_info_list}")

        tb_urls = [tb_url for role_name, _, _, tb_url in role_info_list if
                   role_name == raytf_executor.SIDECAR_TB_ROLE_NAME]
        if tb_urls and len(tb_urls) == 1:
            self.__tb_url = tb_urls[0]

        ray.get([executor.set_tf_cluster_spec.remote(role_info_list) for executor in self.__role_executors_list])

    def __reserved_resources(self, resources):
        resource_bundles = []
        for _, role_resources_dict in resources.items():
            cores, memory_bytes, instances = self.__parse_resources_conf(role_resources_dict)

            resource_bundles.extend(
                [
                    {
                        "CPU": cores,
                        "memory": memory_bytes,
                    }
                    for _ in range(instances)
                ]
            )
        pg = placement_group(resource_bundles, strategy="SPREAD")
        self.__placement_group = pg
        ready, _ = ray.wait([pg.ready()], timeout=self.__resources_reserved_timeout)
        if not ready:
            error_message = f"Failed to get resources, driver exit because of reserving resources timeout."
            self.__logger.error(error_message)
            raise Exception(error_message)

        self.__logger.info(f"Reserved resources list: {placement_group_table(pg)}")

    def __parse_resources_conf(self, role_resources_dict) -> Tuple:
        cores = int(role_resources_dict["cores"]) if role_resources_dict["cores"] else 1
        memory = int(role_resources_dict["memory"]) if role_resources_dict["memory"] else 1
        memory_bytes = memory * 1024 * 1024 * 1024
        instances = int(role_resources_dict["instances"]) if role_resources_dict["instances"] else 1
        return cores, memory_bytes, instances

    def get_tracked_role_size(self, role_executors_list, need_tracked_role_list):
        tracked_executors = [executor for executor in role_executors_list
                             if ray.get(executor.get_role_name.remote()) in need_tracked_role_list]
        return len(tracked_executors)

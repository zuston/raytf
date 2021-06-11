import logging

import ray
from typing import Tuple, List
import random
import time
import json
import os

import log_utils
import tool_utils


@ray.remote(max_retries=0)
class Executor:
    def __init__(self, role_name: str, role_index: int):
        self.__logger = log_utils.get_logger(__name__)
        self.__role_name = role_name
        self.__role_index = role_index
        # 初始化节点地址和 grpc 端口，用来构建 tf cluster spec.
        self.__node_ip = ray._private.services.get_node_ip_address()
        self.__grpc_port = tool_utils.get_reserved_grpc_port()

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

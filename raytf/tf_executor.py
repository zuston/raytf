import ray
from typing import Tuple, List
import json
import os

from raytf import log_utils
from raytf import tool_utils

SIDECAR_TB_ROLE_NAME = "sidecar_tensorboard"


@ray.remote(max_retries=0)
class Executor:
    def __init__(self, role_name: str, role_index: int, event_log_path: str = None):
        self.__logger = log_utils.get_logger(__name__)
        self.__role_name = role_name
        self.__role_index = role_index
        # 初始化节点地址和 grpc 端口，用来构建 tf cluster spec.
        self.__node_ip = ray._private.services.get_node_ip_address()
        self.__grpc_port = tool_utils.get_reserved_grpc_port()

        self.__is_sidecar_tb_enabled = True if role_name == SIDECAR_TB_ROLE_NAME else False
        self.__event_log_path = event_log_path
        if self.__is_sidecar_tb_enabled and not event_log_path:
            raise Exception(f"Sidecar Tensorboard must exist event log path.")
        self.__tb_port = -1 if not self.__is_sidecar_tb_enabled else tool_utils.get_free_port()

    def get_role_info(self) -> Tuple[str, int, str, str]:
        return self.__role_name, self.__role_index, f"{self.__node_ip}:{self.__grpc_port}", f"{self.__node_ip}:{self.__tb_port}"

    def set_tf_cluster_spec(self, spec_info: List[Tuple[str, int, str]]):
        os.environ['ROLE_NAME'] = self.__role_name
        os.environ['ROLE_INDEX'] = str(self.__role_index)

        if self.__is_sidecar_tb_enabled:
            return

        spec_tmp_dict = {}
        for (role, _, hostip, _) in spec_info:
            if role == SIDECAR_TB_ROLE_NAME:
                continue
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
        if self.__is_sidecar_tb_enabled:
            self.__start_sidecar_tensorboard()
        else:
            model_process(args)
        self.__logger.info(f"[{self.__role_name}][{self.__role_index}], Finished")
        return self.__role_name, self.__role_index

    def __start_sidecar_tensorboard(self):
        from tensorboard.main import run_main
        import sys
        import re
        sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
        sys.argv = [sys.argv[0], "--logdir", self.__event_log_path, "--port", str(self.__tb_port),
                    "--host", self.__node_ip]
        self.__logger.info(f"Starting tensorboard with args: {sys.argv}")
        run_main()


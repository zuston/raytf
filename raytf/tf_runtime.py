import json
from typing import Dict, List, Tuple

from raytf import log_utils

from raytf.raytf_executor import SIDECAR_TB_ROLE_NAME
from raytf.runtime import Runtime


class TFRuntime(Runtime):
    def __init__(self):
        self.__logger = log_utils.get_logger(__name__)

    def get_tracked_role_list(self) -> List[str]:
        return ['worker', 'chief', 'evaluator']

    def construct_cluster_spec(self, role_name: str, role_index: int,
                               spec_info: List[Tuple[str, int, str]]) -> Dict[str, str]:
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
                "type": role_name,
                "index": role_index
            }
        }
        cluster_spec_json = json.dumps(tf_spec_dict)
        self.__logger.info(f"[{role_name}][{role_index}], tf cluster spec: {cluster_spec_json}")

        return {
            'TF_GRPC_REUSE_PORT': "true",
            'TF_CONFIG': cluster_spec_json
        }
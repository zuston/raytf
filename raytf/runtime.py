from abc import ABC, abstractmethod
from typing import Tuple, List, Dict


class Runtime(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def construct_cluster_spec(self, role_name: str, role_index: int,
                               spec_info: List[Tuple[str, int, str]]) -> Dict[str, str]:
        """
        """

    @abstractmethod
    def get_tracked_role_list(self) -> List[str]:
        """
        """
import ray

from model import process
from tf_cluster import TensorflowCluster

if __name__ == '__main__':
    # ray.init(address='auto')

    # 准备环境，进行训练
    tf_cluster = TensorflowCluster.build(resources=
        {
            "ps": {"cores": "2", "memory": "2", "gpu": "2", "instances": "1"},
            "worker": {"cores": "2", "memory": "2", "gpu": "2", "instances": "1"},
            "chief": {"cores": "2", "memory": "2", "gpu": "2", "instances": "1"}
        }
    )
    tf_cluster.start(model_process=process, args=None)

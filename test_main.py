import ray

from model import process
from tf_cluster_driver import TensorflowCluster

if __name__ == '__main__':

    # ray.init()

    tf_cluster = TensorflowCluster.build(resources=
        {
            "ps": {"cores": "2", "memory": "2", "gpu": "2", "instances": "1"},
            "worker": {"cores": "2", "memory": "2", "gpu": "2", "instances": "1"},
            "chief": {"cores": "2", "memory": "2", "gpu": "2", "instances": "1"}
        },
        event_log="/tmp/opal/4"
    )
    tf_cluster.start(model_process=process, args=None)

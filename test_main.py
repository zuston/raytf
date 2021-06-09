import ray

from model import process
from tf_cluster_driver import TensorflowCluster

if __name__ == '__main__':
    # jobconf = ray.job_config.JobConfig()
    # jobconf.code_search_path = "/"
    # ray.init(address='auto', job_config=jobconf)
    ray.init()

    # 准备环境，进行训练
    tf_cluster = TensorflowCluster.build(resources=
        {
            "ps": {"cores": "2", "memory": "2", "gpu": "2", "instances": "1"},
            "worker": {"cores": "2", "memory": "2", "gpu": "2", "instances": "1"},
            "chief": {"cores": "2", "memory": "2", "gpu": "2", "instances": "1"}
        }
    )
    tf_cluster.start(model_process=process, args=None)

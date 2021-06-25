import ray

from tests.sample_tf_estimator_model import process
from tests.sample_tf_keras_tony_model import tony_keras_process
from tf_cluster_driver import TensorflowCluster
import pytest
import sys


@pytest.fixture(scope="function")
def init_cluster():
    if not ray.is_initialized():
        ray.init()


def test_keras_multiworker_strategy(init_cluster):
    tf_cluster = TensorflowCluster.build(resources=
    {
        "worker": {"cores": "2", "memory": "1", "gpu": "2", "instances": "1"},
    }
    )
    tf_cluster.start(model_process=tony_keras_process, args=None)


def test_estimator_ps_strategy(init_cluster):
    tf_cluster = TensorflowCluster.build(resources=
    {
        "ps": {"cores": "1", "memory": "1", "gpu": "2", "instances": "1"},
        "worker": {"cores": "1", "memory": "1", "gpu": "2", "instances": "1"},
        "chief": {"cores": "1", "memory": "1", "gpu": "2", "instances": "1"}
    }
    )
    tf_cluster.start(model_process=process, args=None)


if __name__ == '__main__':
    sys.exit(pytest.main(["-v", __file__]))

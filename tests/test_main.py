import ray

from tests.sample_tf_estimator_model import process
from tests.sample_tf_keras_tony_model import tony_keras_process
from tf_cluster_driver import TensorflowCluster
import pytest
import sys
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group
)


@pytest.fixture(scope="function")
def init_cluster():
    if not ray.is_initialized():
        ray.init(num_cpus=4)


@pytest.mark.skip()
def test_keras_multiworker_strategy(init_cluster):
    tf_cluster = TensorflowCluster.build(resources=
    {
        "worker": {"cores": "2", "memory": "1", "gpu": "2", "instances": "1"},
    }
    )
    tf_cluster.start(model_process=tony_keras_process, args=None)


@pytest.mark.skip()
def test_estimator_ps_strategy(init_cluster):
    tf_cluster = TensorflowCluster.build(resources=
    {
        "ps": {"cores": "1", "memory": "1", "gpu": "2", "instances": "1"},
        "worker": {"cores": "1", "memory": "1", "gpu": "2", "instances": "1"},
        "chief": {"cores": "1", "memory": "1", "gpu": "2", "instances": "1"}
    }
    )
    tf_cluster.start(model_process=process, args=None)

@pytest.mark.skip()
def test_model_without_resources_will_exit(init_cluster):
    tf_cluster = TensorflowCluster.build(
        resources=
        {
            "ps": {"cores": "2", "memory": "1", "gpu": "2", "instances": "1"},
            "worker": {"cores": "2", "memory": "1", "gpu": "2", "instances": "1"},
            "chief": {"cores": "1", "memory": "1", "gpu": "2", "instances": "1"}
        },
        resources_reserved_timeout=10
    )
    try:
        tf_cluster.start(model_process=process, args=None)
        pytest.fail("Should exit. And raise exception.")
    except Exception as e:
        pass


# 测试下 placement group 设置资源组，来实现 gang scheduler
def test_gang_scheduler(init_cluster):
    pg = placement_group(
        [
            {"CPU": 2},
            {"CPU": 1},
            {"CPU": 1}
        ]
    )
    ready, _ = ray.wait([pg.ready()])
    if not ready:
        pytest.fail("Should get the resources")

    @ray.remote
    def out():
        print("Hello world")
        return

    ray.get(out.options(
        placement_group=pg,
        num_cpus=2
    ).remote())

if __name__ == '__main__':
    sys.exit(pytest.main(["-v", __file__, "--capture=no"]))

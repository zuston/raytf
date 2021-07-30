Tensorflow Cluster on Ray
-------------------------

How to Use?
~~~~~~~~~~~

.. code:: python


        tf_cluster = TensorflowCluster.build(resources=
            {
                "ps": {"cores": "2", "memory": "2", "gpu": "2", "instances": "1"},
                "worker": {"cores": "2", "memory": "2", "gpu": "2", "instances": "1"},
                "chief": {"cores": "2", "memory": "2", "gpu": "2", "instances": "1"}
            },
            event_log="/tmp/opal/4"
        )
        tf_cluster.start(model_process=process, args=None)

This training code will be attached to the existed perm-Ray cluster. If
you want to debug, you can use ``ray.init()`` to init Ray cluster in
local.

When you specify the event\_log in tf builder, sidecar tensorboard will
be started on one worker.

How to build
~~~~~~~~~~~~

[Requirement] python -m pip install twine

1. python setup.py bdist\_wheel --universal
2. python -m pip install xxxxxx.whl
3. twine upload dist/*

Tips
~~~~

1. To solve the problem of Python module importing on Ray perm-cluster,
   this project must use Ray 1.5+ version, refer to this
   RFC(https://github.com/ray-project/ray/issues/14019)
2. This project is only be tested by Tensorflow estimator training

Distributed Deep Learning Framework on Ray
--------------------------------------------------

The raytf framework provides a simple interface to support distributed training on ray,
including tensorflow/pytorch/mxnet. Now tensorflow has been supported,
others will be included in later.

Quick Start
~~~~~~~~~~~
Only tested under Python3.6 version

1. Install the latest ray version: ``pip install ray``
2. Install the latest raytf: ``pip install raytf``
3. Git clone this project: ``git clone https://github.com/zuston/raytf.git``
4. Enter the example folder and execute the python script file, like the following command.

.. code:: bash

        cd raytf
        cd example
        python mnist.py


How to Use
~~~~~~~~~~~

.. code:: python

        from raytf.raytf_driver import Driver
        # When you using it in local single machine
        # ray.init()
        tf_cluster = Driver.build(resources=
            {
                'ps': {'cores': 2, 'memory': 2, 'gpu': 2, 'instances': 2},
                'worker': {'cores': 2, 'memory': 2, 'gpu': 2, 'instances': 6},
                'chief': {'cores': 2, 'memory': 2, 'gpu': 2, 'instances': 1}
            },
            event_log='/tmp/opal/4',
            resources_allocation_timeout=10
        )
        tf_cluster.start(model_process=process, args=None)

This training code will be attached to the existed perm-Ray cluster. If
you want to debug, you can use ``ray.init()`` to init Ray cluster in
local.

When you specify the event\_log in tf builder, sidecar tensorboard will
be started on one worker.

GANG scheduler has been supported.Besides raytf provides the
configuration of timeout for waiting resources
which is shown in above code. The ``resources_allocation_timeout`` unit is sec

How to build and deploy
~~~~~~~~~~~~~~~~~~~~~~~

<Requirement> ``python -m pip install twine``

1. ``python setup.py bdist\_wheel --universal``
2. ``python -m pip install xxxxxx.whl``
3. ``twine upload dist/*``

Tips
~~~~

1. To solve the problem of Python module importing on Ray perm-cluster,
   this project must use Ray 1.5+ version, refer to this
   RFC(https://github.com/ray-project/ray/issues/14019)
2. This project is only be tested by Tensorflow estimator training


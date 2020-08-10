Frequently Asked Questions
==========================

Can I use stream anomaly models for batch anomaly detection tasks?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The stream anomaly detection models can be used for batch anomaly detection tasks. Yet, the models are mainly designed for the tasks when a model sees an instance only once and removes from memory. Since models cannot always access all data, their performance may suffer. However, they are better fit to real-world problems when data arrives as time passes. You may also see `PyOD library <https://pyod.readthedocs.io/en/latest/>`_ for batch learning models.

Can I use batch learning model for stream learning tasks?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Batch anomaly detection models from `PyOD library <https://pyod.readthedocs.io/en/latest/>`_ can be used by reference windowing via :class:`pysad.models.reference_window_model` or fitting to initial instances via :class:`pysad.models.one_fit_model`. An example of integration is provided in `Github <https://github.com/selimfirat/pysad/blob/master/examples/example_pyod_integration.py>`_.

How can I contribute to this project?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please see :ref:`contributing` for details.

I found a bug. What should I do?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You may contribute to this framework by sharing these bugs or feature ideas for improvements. You should open an issue on our `Github repository <https://github.com/selimfirat/pysad>`_. You are also welcome to contribute via opening a pull request. Please see :ref:`contributing` for more information.
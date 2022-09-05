.. _viz:

CDR Visualizer
==============

The CDR Visualizer supports interactive plotting of CDR estimates in your web browser.
To run it, you will first need to install the ``dash`` library using either Anaconda or pip, e.g.::

    pip install dash

Once dash is installed, you can visualize any trained model by running::

    python -m cdr.viz.app <PATH/TO/CONFIG> -m <MODEL_NAME>

where ``<PATH/TO/CONFIG>`` and ``<MODEL_NAME>`` are respectively replaced by the path to your config file and the name specified in the config for the model you want to visualize.
CDR Visualizer is a useful complement to the static plotting tools provided by ``cdr.bin.plot`` because it permits flexible, fast exploration of the solution space recovered by the model.
This tool is currently in early development and many planned features are yet to be implemented.
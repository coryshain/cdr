.. _convergence:

Diagnosing convergence
======================

Intuitively, CDR models have converged when have ceased to improve with additional training.
Programmatically detecting this condition is difficult because of noise in the parameter iterates introduced by stochastic optimization.
The level of noise can vary as a function of the data, data size, model structure, learning rate, and optimizer choice.
This package provides automatic convergence diagnostics that declare convergence when the loss is uncorrelated with training time within a tolerance.
The default settings for these diagnostics are designed to be general purpose but may not be appropriate for all model and datasets.
Therefore, it is always a good idea to visually check the learning curves in Tensorboard in order to confirm the automatic diagnosis of (non-)convergence.
To do so, run the following from the command line::

    python -m tensorboard.main --logdir=<path_to_model_directory>

Then open the displayed URL in your web browser to visualize the learning curves over the whole parameter space.
Fixed effects estimates are displayed under the "Scalars" tab, and random effects estimates are displayed under the "Distributions" tab.
If the model threw a convergence warning but all estimates visually appear to have flatlined, the diagnostics may have been too conservative and you can consider the model converged.
Conversely, if the model did not throw a convergence warning but one or more estimates still appear to be moving in a particular direction, the diagnostics may have been too permissive and you should not consider the model converged.
In this case, you will need to revise the convergence diagnostics (e.g. set a lower value for **convergence_tolerance** in the config file) and continue training.

The convergence diagnostics have a material impact on training in that they are a sufficient stopping condition.
If either the model has been diagnosed as converged or reached the limit imposed by the **n_iter** parameter, training will automatically stop.
Thus, it may be necessary to revise the convergence parameters at the outset of or during training.
To consult the defaults used for these convergence checking parameters, see :ref:`config`.
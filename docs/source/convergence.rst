.. _convergence:

Diagnosing convergence
======================

Intuitively, DTSR models have converged when all their parameter estimates have ceased to move with additional training epochs.
Programmatically detecting this condition is difficult because of noise in the parameter iterates introduced by stochastic optimization.
The level of noise can vary as a function of the data, data size, model structure, learning rate, and optimizer choice.
Thus, while this package provides programmatic convergence diagnostics, it is important to recognize that the convergence parameters may be too conservative or permissive for your particular model.
It is always a good idea to visually check the learning curves in Tensorboard in order to confirm the automatic diagnosis of (non-)convergence.
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
In general, factors that increase noise in the stochastic optimization (smaller data, more complex models, and higher learning rates) may require more permissive convergence criteria.


How convergence diagnosis is implemented in this package
--------------------------------------------------------

Convergence diagnosis in DTSR depends on estimating the first and second derivatives of all parameters with respect to training time.
If the largest first derivative, along with the second derivative at that point, is within some tolerance of zero, the model can be considered to have converged at that tolerance level because the parameters are not changing and parameter change is not accelerating.
To estimate the first derivative, the system incrementally saves iterates of the model parameters and computes the slope of the regression line to these iterates for each parameter as a function of training epoch.
To estimate the second derivative, the system incrementally saves its first derivative estimates as computed above, then finds the slope of the regression line to the first derivatives as a function of training epoch.
The number of iterates retained and the frequency with which they are recorded are governed by the **convergence_n_iterates** and **convergence_check_freq** parameters, respectively.
The receptive field of the derivative estimates is ``convergence_n_iterates * convergence_check_freq`` training epochs.
Increasing this receptive field increases smoothing on the derivative estimates, which has advantages and drawbacks.
The larger the receptive field, the less sensitive the derivative estimates are to local fluctuations, but the more post-convergence iterations necessary to detect convergence, increasing training time.
Training stops automatically when the derivatives estimated above both fall below **convergence_tolerance** (or when **n_iter** is reached, whichever happens first).

This implementation of convergence checking can also impose substantial memory overhead, because ``2 * convergence_n_iterates + 1`` copies of the model (or its first derivatives) must be retained in memory.
In many cases this does not pose a problem, since the model is often much smaller than the data.
But for very large models it can create memory bottlenecks, which can be resolved by reducing **convergence_n_iterates**.
To turn off convergence checking (and its attendant speed and memory costs), simply set **convergence_n_iterates** and/or **convergence_check_freq** to ``0`` or ``None``.
In this case, visual diagnosis of convergence is necessary.

To consult the defaults used for these convergence checking parameters, see :ref:`config`.
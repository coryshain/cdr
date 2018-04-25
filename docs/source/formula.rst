.. _formula:

DTSR Model Formulae
===================

This package constructs DTSR models from ``R``-style formula strings defining the model structure.
A DTSR formula has the following template:

``RESPONSE ~ FIXED_EFFECTS + RANDOM_EFFECTS``

The left-hand side (LHS) of the formula contains the name a single (possibly transformed) variable from the input data table, and the right-hand side (RHS) contains fixed and random effects, each of which must consist exclusively of intercept terms and/or convolutional terms.

A convolutional term is defined using the syntax ``C(..., IRF_FAMILY())``, where ``...`` is replaced by names of predictors contained in the input data.
For example, to define a ``Gamma`` convolution of predictor ``A``, the expression ``C(A, Gamma())`` is added to the RHS.
Separate terms are delimited by ``+``.
For example, to add a Gaussian convolution of predictor ``B``, the RHS above becomes ``C(A, Gamma()) + C(B, Normal())``.

The currently supported IRF families are:

- ``DiracDelta``: Stick function

  - Parameters: None
  - Definition: :math:`1` at :math:`x=0`, :math:`0` otherwise

- ``Exp``: PDF of exponential distribution

  - Parameters: :math:`\lambda` (rate)
  - Definition: :math:`\lambda e^{-\lambda x}`

- ``Gamma``: PDF of gamma distribution

  - Parameters: :math:`k` (shape), :math:`\theta` (rate)
  - Definition: :math:`\frac{x^{k-1}e^{-\frac{x}{\theta}}{\theta^k\Gamma(k)}`

- ``ShiftedGamma``: PDF of gamma distribution with support starting at :math:`0 - \delta`

  - Parameters: :math:`k` (shape), :math:`\theta` (rate), :math:`\delta` (shift, strictly negative)
  - Definition: :math:`\frac{(x - \delta)^{k-1}e^{-\frac{x - \delta}{\theta}}{\theta^k\Gamma(k)}`

- ``GammaKgt1``: PDF of gamma distribution, :math:`k > 1` (enforces rising-then-falling shape)

  - Parameters: :math:`k` (shape), :math:`\theta` (rate)
  - Definition: :math:`\\frac{x^{k-1}e^{-\\frac{x}{\theta}}{\theta^k\Gamma(k)}`

- ``ShiftedGammaKgt1``: PDF of gamma distribution with support starting at :math:`0 - \delta`, :math:`k > 1` (enforces rising-then-falling shape)

  - Parameters: :math:`k` (shape), :math:`\theta` (rate), :math:`\delta` (shift, strictly negative)
  - Definition: :math:`\frac{(x - \delta)^{k-1}e^{-\frac{x - \delta}{\theta}}{\theta^k\Gamma(k)}`

- ``Normal``
- ``SkewNormal``
- ``EMG``
- ``BetaPrime``
- ``ShiftedBetaPrime``


For convenience, the ``C()`` function distributes the impulse response family over multiple ``+``-delimited terms in its first argument.
Therefore, the following two expressions are equivalent:

``C(A + B, Gamma())``
``C(A, Gamma()) + C(B, Gamma())``

As in ``R``, interaction terms are designated with ``:``, as in ``C(A:B, Gamma())``, and cross-product interactions can be expressed using Python's power notation ``**<INT>``.
For example, ``(A + B + C)**3`` adds all first, second, and third order interactions, expanding out as:

``A + B + C + A:B + B:C + A:C + A:B:C``

As above, IRF distribute across the expansion of interaction terms, such that the following expressions are equivalent:

``C((A + B + C)**3, Gamma())``
``C(A, Gamma()) + C(B, Gamma()) + C(C, Gamma()) + C(A:B, Gamma()) + C(B:C, Gamma()) + C(A:C, Gamma()) + C(A:B:C, Gamma())``

Unlike ``R``, categorical variables are not yet handled automatically in DTSR.
However, they can be considered simply by adding binary vectors for each of the :math:`n-1` levels of the variable to the input data.

Note that the variable expansions described above add `separate` IRF for each term in the expansion.
For example, ``C(A + B, Gamma())`` adds `two` distinct Gamma IRF parameterizations to the model, one for each predictor.

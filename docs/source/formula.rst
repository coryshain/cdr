.. _formula:

DTSR Model Formulae
===================



Basic Overview
--------------

This package constructs DTSR models from **R**-style formula strings defining the model structure.
A DTSR formula has the following template::

    RESPONSE ~ FIXED_EFFECTS + RANDOM_EFFECTS

The left-hand side (LHS) of the formula contains the name a single (possibly transformed) variable from the input data table, and the right-hand side (RHS) contains fixed and random effects, each of which must consist exclusively of intercept terms and/or convolutional terms.
Intercept terms can be added by including ``1`` in the RHS and removed by including ``0`` in the RHS.
If neither of these appears in the RHS, an intercept is added by default.



Defining an Impulse Response Function (IRF)
-------------------------------------------

A convolutional term is defined using the syntax ``C(..., IRF_FAMILY())``, where ``...`` is replaced by names of predictors contained in the input data.
For example, to define a Gamma convolution of predictor ``A``, the expression ``C(A, Gamma())`` is added to the RHS.
Separate terms are delimited by ``+``.
For example, to add a Gaussian convolution of predictor ``B``, the RHS above becomes ``C(A, Gamma()) + C(B, Normal())``.



Supported Parametric IRF
-------------

The currently supported parametric IRF families are:

- ``DiracDelta``: Stick function (equivalent to a predictor in linear regression)

  - Parameters:

    - None

  - Definition: :math:`1` at :math:`x=0`, :math:`0` otherwise

- ``Exp``: PDF of exponential distribution

  - Parameters:

    - :math:`\beta > 0`: ``beta`` (rate)

  - Definition: :math:`\beta e^{-\beta x}`

- ``Gamma``: PDF of gamma distribution

  - Parameters:

    - :math:`\alpha > 0`: ``alpha`` (shape)
    - :math:`\beta > 0`: ``beta`` (rate)

  - Definition: :math:`\frac{\beta^{\alpha}x^{\alpha-1}e^{-\frac{x}{\beta}}}{\Gamma(\alpha)}`

- ``ShiftedGamma``: PDF of gamma distribution with support starting at :math:`0 + \delta`

  - Parameters:

    - :math:`\alpha > 0`: ``alpha`` (shape)
    - :math:`\beta > 0`: ``beta`` (rate)
    - :math:`\delta < 0`: ``delta`` (shift)

  - Definition: :math:`\frac{\beta^{\alpha}(x - \delta)^{\alpha-1}e^{-\frac{x - \delta}{\beta}}}{\Gamma(\alpha)}`

- ``GammaShapeGT1``: PDF of gamma distribution, :math:`\alpha > 1` (enforces rising-then-falling shape)

  - Parameters:

    - :math:`\alpha > 1`: ``alpha`` (shape)
    - :math:`\beta > 1`: ``beta`` (rate)

  - Definition: :math:`\frac{\beta^{\alpha}x^{\alpha-1}e^{-\frac{x}{\beta}}}{\Gamma(\alpha)}`

- ``ShiftedGammaShapeGT1``: PDF of gamma distribution with support starting at :math:`0 + \delta`, :math:`\alpha > 1` (enforces rising-then-falling shape)

  - Parameters:

    - :math:`\alpha > 1`: ``alpha`` (shape)
    - :math:`\beta > 0`: ``beta`` (rate)
    - :math:`\delta < 0`: ``delta`` (shift)

  - Definition: :math:`\frac{\beta^{\alpha}(x - \delta)^{\alpha-1}e^{-\frac{x - \delta}{\beta}}}{\Gamma(\alpha)}`

- ``Normal``: PDF of Gaussian (normal) distribution

  - Parameters:

    - :math:`\mu`: ``mu`` (mean)
    - :math:`\sigma > 0`: ``sigma`` (standard deviation)

  - Definition: :math:`\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x - \mu) ^ 2}{2 \sigma ^ 2}}`

- ``SkewNormal``: PDF of SkewNormal distribution (normal distribution augmented with left/right skew parameter)

  - Parameters:

    - :math:`\mu` (mean)
    - :math:`\sigma` (standard deviation)
    - :math:`\alpha` (skew)

  - Definition: Let :math:`\phi` and :math:`\Phi` denote the PDF and CDF (respectively) of the standard normal distribution.
    Then the SkewNormal distribution is:
    :math:`\frac{2}{\sigma} \phi\left(\frac{x-\mu}{\sigma}\right) \Phi(\alpha \frac{x-\mu}{\sigma})`

- ``EMG``: PDF of exponentially modified gaussian distribution (convolution of a normal with an exponential distribution, can be right-skewed)

  - Parameters:

    - :math:`\mu`: ``mu`` (mean)
    - :math:`\sigma > 0`: ``sigma`` (standard deviation)
    - :math:`\beta > 0`: ``beta`` (rate)

  - Definition: :math:`\frac{\beta}{2}e^{\frac{\beta}{2}\left(2\mu + \beta \sigma^2 - 2x \right)} \mathrm{erfc} \left(\frac{m + \beta \sigma ^2 - x}{\sqrt{2}\sigma}\right)`, where :math:`\mathrm{erfc}(x) = \frac{2}{\sqrt{\pi}}\int_x^{\infty} e^{-t^2}dt`.

- ``BetaPrime``: PDF of BetaPrime (inverted beta) distribution

  - Parameters:

    - :math:`\alpha > 0`: ``alpha`` (shape)
    - :math:`\beta > 0`: ``beta`` (shape)

  - Definition: :math:`\frac{x^{\alpha - 1}(1 + x)^{-\alpha - \beta}}{B(\alpha, \beta)}`

- ``ShiftedBetaPrime``: PDF of BetaPrime (inverted beta) distribution with support starting at :math:`0 + \delta`

  - Parameters:

    - :math:`\alpha > 0`: ``alpha`` (shape)
    - :math:`\beta > 0`: ``beta`` (shape)
    - :math:`\delta < 0`: ``delta`` (shift)

  - Definition: :math:`\frac{(x-\delta)^{\alpha - 1}(1 + (x - \delta))^{-\alpha - \beta}}{B(\alpha, \beta)}`

- ``HRFSingleGamma``: Single-gamma hemodynamic response function (fMRI). Identical to ``GammaShapeGT1`` except in its initial parameter values, which are inherited from the peak response model of the canonical HRF in SPM (:math:`\alpha = 6` and :math:`\beta = 1`)

  - Parameters:

    - :math:`\alpha > 0`: ``alpha`` (shape)
    - :math:`\beta > 0`: ``beta`` (rate)

  - Definition: :math:`\frac{\beta^{\alpha}x^{\alpha-1}e^{-\frac{x}{\beta}}}{\Gamma(\alpha)}`

- ``HRFDoubleGamma``: Double-gamma hemodynamic response function (fMRI, peak response with undershoot) with constraints against implausible undershoot models. Specifically, the shape parameter :math:`\alpha` of the undershoot is constrained to be at least as large as the shape parameter of the peak response, the rate parameter :math:`\beta` is tied between the peak response and the undershoot, and the constant coefficient :math:`c` is constrained to be between 0 and 1, ensuring that the undershoot will be both negative and smaller in magnitude than the peak response. In some applications it can also be useful to simply leave the shape of the undershoot and the coefficient of the undershoot untrained, resulting in a two-parameter model with a global shape and rate. This can be accomplished using the keyword specification ``trainable=[alpha, beta]`` in the IRF call in the model formula.

  - Parameters:

    - :math:`\alpha_1 > 0`: ``alpha_main`` (peak response shape)
    - :math:`\beta > 0`: ``beta`` (rate)
    - :math:`\alpha_2 > 0`: ``alpha_undershoot_offset`` (offset for undershoot shape)
    - :math:`0 < c < 1`: ``c`` (undershoot coefficient)

  - Definition: :math:`\frac{\beta^{\alpha_1}x^{\alpha_1-1}e^{-\frac{x}{\beta}}}{\Gamma(\alpha_1)} - c\frac{\beta^{\alpha_1 + \alpha_2}x^{\alpha_1 + \alpha_2 -1}e^{-\frac{x}{\beta}}}{\Gamma(\alpha_1 + \alpha_2)}`

- ``HRFDoubleGammaUnconstrained``: Double-gamma hemodynamic response function (fMRI, peak response with undershoot) without constraints on the undershoot models.

  - Parameters:

    - :math:`\alpha_1 > 0`: ``alpha_main`` (peak response shape)
    - :math:`\beta_1 > 0`: ``beta_main`` (peak response rate)
    - :math:`\alpha_2 > 0`: ``alpha_undershoot`` (undershoot shape)
    - :math:`\beta_2 > 0`: ``beta_undershoot`` (undershoot rate)
    - :math:`c`: ``c`` (undershoot coefficient)

  - Definition: :math:`\frac{\beta^{\alpha_1}x^{\alpha_1-1}e^{-\frac{x}{\beta_1}}}{\Gamma(\alpha_1)} - c\frac{\beta^{\alpha_2}x^{\alpha_2 - 1}e^{-\frac{x}{\beta_2}}}{\Gamma(\alpha_2)}`



Spline IRF
----------

DTSR also supports non-parametric IRF in the form of spline functions.
Instead of a parametric IRF kernel, the model is supplied with control points (knots) that define a smooth function which can be moved around the x/y plane.
The advantage of spline IRF is that they do not require precommitment to a particular functional form for the IRF.
The disadvantage is that fitting them is much more computationally expensive because computing the spline function between the control points requires matrix inversion.

The splines themselves have a number of free parameters which are specified by the name of the spline in the IRF call of the model formula.
The syntax for a spline IRF kernal is as follows::

    S(o([0-9]+))?(b([0-9]+))?(l([0-9]+))?(p([0-9]+))?(i([0-1]))?

This is a string representation of a function call ``S`` with optional keyword arguments ``o``, ``b``, ``l``, ``p``, and ``i``, in that order.

The keyword arguments are defined as follows:

  - **o** (order): ``int``, the order of the spline. Order 1 is linear interpolation, order 2 is a thin-plate spline, order 3 is a cubic spline, etc. **Default**: 2.
  - **b** (bases): ``int``, number of bases (control points). **Default**: 10.
  - **l** (roughness penalty): ``int``, digits following the decimal representing the roughness penalty (regularization against wiggliness). For example, ``l01`` specifies a roughness penalty of 0.01. **Default**: 001.
  - **p** (spacing power): ``int``, power to use for initial spacing of control points in time between 0 and the maximum time offset attested in the training data. If 1, control points will be initialized as evenly spaced. If 2, control points will be quadratically spaced, etc. Initially concentrating more control points toward smaller time offsets is motivated in most cases by the fact that (1) many real-world IRF have more complex dynamics closer to the time of the impulse and (2) most datasets will contain more training data for smaller time offsets than longer ones, possibly resulting in decreasing precision of the IRF estimate at long latencies. **Default**: 1.
  - **i** (instantaneous response): ``int`` (0 or 1), whether to allow an instantaneous response. If 0, the response at time 0 is forced to be 0. **Default**: 1.



IRF Composition
---------------

In some cases it may be desirable to decompose the response into multiple convolutions of an impulse.
For example, it is reasonable to suppose that the BOLD response in fMRI may underlyingly consist of 2 convolutional responses: a **neural response** may convolve the impulse into a timecourse of neural activation which is then convolved with a **hemodynamic response** into a BOLD signal.
In this case, it would be desirable to be able to model the BOLD response as a composition of neural and hemodynamic responses.

Exact parametric composition of IRF is not possible in the general case because many pairs of IRF do not have a tractable analytical convolution.
Instead, the DTSR package uses a discrete approximation to the continuous integral of composed IRF by (1) computing the value of each IRF for some number of interpolation points, (2) computing their convolution via FFT, and (3) rescaling by the temporal distance between interpolation points.
The number of interpolation points is defined by the model's **n_interp** initialization parameter.

To compose IRF in a model, simply insert one IRF call into the first argument position of another IRF call.
For example, the following first convolves impulse ``A`` with a normal IRF and then convolves this convolved response with an exponential IRF::

    C(A, Exp(Normal()))

Because convolution has the associative property, the order of composition does not matter, and the above is equivalent to::

    C(A, Normal(Exp()))

The advantage of IRF composition is that it affords the possibility of discovering the structure of latent responses that are not directly observable in the measured response, as in the example described above.
The disadvantage is that it is much more computationally expensive due to the interpolation and FFT steps required.

Care must also be taken when using IRF composition to avoid constructing unidentifiable models.
For example, the convolution of two Gaussians :math:`N(\mu_1, \sigma_1^2)` and :math:`N(\mu_2, \sigma_2^2)` is known to be :math:`N(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)`.
As a result, the following composed IRF has infinitely many solutions, and the resulting model is unidentifiable::

    C(A, Normal(Normal()))

DTSR is not able to recognize and flag identifiability problems and it will happily find a solution to such a model, disguising the fact that there are infinitely many other optimal solutions.
It is up to the user to think carefully about whether the model structure could introduce such problems.
For example, in the BOLD example discussed above, the neural response is predictor-specific while the hemodynamic response is predictor-independent given the neural response.
The two responses can thus be separated via parameter tying of the hemodynamic response portion (see below), requiring all predictors to share a single hemodynamic response and forcing predictor-level variation into the neural response alone.



Automatic Term Expansion
------------------------

For convenience, the ``C()`` function distributes the impulse response family over multiple ``+``-delimited terms in its first argument.
Therefore, the following two expressions are equivalent::

    C(A + B, Gamma())
    C(A, Gamma()) + C(B, Gamma())

As in **R**, interaction terms are designated with ``:``, as in ``C(A:B, Gamma())``, and cross-product interactions can be expressed using Python's power notation ``**<INT>``.
For example, ``(A + B + C)**3`` adds all first, second, and third order interactions, expanding out as::

    A + B + C + A:B + B:C + A:C + A:B:C

As above, IRF distribute across the expansion of interaction terms, such that the following expressions are equivalent::

    C((A + B + C)**3, Gamma())
    C(A, Gamma()) + C(B, Gamma()) + C(C, Gamma()) + C(A:B, Gamma()) + C(B:C, Gamma()) + C(A:C, Gamma()) + C(A:B:C, Gamma())

Unlike **R**, categorical variables are not yet handled automatically in DTSR.
However, they can be included simply by adding binary indicator vectors for each of :math:`n-1` of the levels of the variable to the input data as a preprocessing step, then defining the model in terms of the binary indicators.

Note that the term expansions described above add `separate` IRF for each term in the expansion.
For example, ``C(A + B, Gamma())`` adds two distinct Gamma IRF parameterizations to the model, one for each predictor.
It is also possible to tie IRF between predictor variables (details below).

Note also that (unlike **R**) redundant terms are **not** automatically collapsed, so care must be taken to ensure that no duplicate terms are produced via term expansion.



Random Effects
--------------

Random effects in DTSR are specified using the following syntax::

    (RANDOM_TERMS | GROUPING_FACTOR)

where ``RANDOM_TERMS`` are terms as they would appear in the RHS of the model described above and ``GROUPING_FACTOR`` is the name of a categorical variable in the input that is used to define the random effect (e.g. a vector of ID's of human subjects).
As in the case of fixed effects, a random intercept is automatically added unless ``0`` appears among the random terms.
Mixed models are constructed simply by adding random effects to fixed effects in the RHS of the formula.
For example, to construct a mixed model with a fixed and by-subject random coefficient for a Gaussian IRF for predictor ``A`` along with a random intercept by subject, the following RHS would be used::

    C(A, Normal()) + (C(A, Normal()) | subject)

IRF in random effects statements are treated as tied to any corresponding fixed effects unless explicitly distinguished by distinct IRF ID's (see section below on parameter tying).

The above formula uses a single parameterization for the Gaussian IRF and fits by-subject coefficients for it.
However it is also possible to fit by-subject IRF parameterizations.
This can be accomplished by adding ``ran=T`` to the IRF call, as shown below::

    C(A, Normal()) + (C(A, Normal(ran=T)) | subject)

This formula will fit separate coefficients `and` IRF shapes for this predictor for each subject.

An important complication in fitting mixed models with DTSR is that the relevant grouping factor is determined by the current `regression target`, not the properties of the independent variable observations in the series history.
This means that random effects are only guaranteed to be meaningful when fit using grouping factors that are constant for the entire series (e.g. the ID of the human subject completing the experiment).
Random effects fit for grouping factors that vary during the experiment should therefore be avoided unless they are intercept terms only, which are not affected by the temporal convolution.



Parameter Initialization
---------------
IRF parameters can be initialized for a given convolutional term by specifying their initial values in the IRF call, using the parameter name as the keyword (see supported IRF and their associated parameters above).
For example, to initialize a Gamma IRF with :math:`\alpha = 2` and :math:`\beta = 2` for predictor ``A``, use the following call::

    C(A, Gamma(alpha=2, beta=2))

These values will serve as initializations in both DTSRMLE and DTSRBayes, and in DTSRBayes they will additionally serve as the mean of the prior distribution for that parameter.
If no initialization is specified, defaults will be used.
These defaults are not guaranteed to be plausible for your particular application and may have a detrimental impact on training.
Therefore, it is generally a good idea to think carefully in advance about what kinds of IRF shapes are `a priori` reasonable and choose initializations in that range.

Note that the initialization values are on the constrained space, so make sure to respect the constraints when choosing them.
For example, :math:`\alpha` of the Gamma distribution is constrained to be > 0, so an initial :math:`\alpha` of <=0 will result in incorrect behavior.
However, keep in mind that for DTSRBayes, prior variances are necessarily on the unconstrained space and get squashed by the constraint function, so choosing initializations that are very close to constraint boundaries can indirectly tighten the prior.
For example, choosing an initialization :math:`\alpha = 0.001` for the Gamma distribution will result in a much tighter prior around small values of :math:`\alpha`.

Initializations for irrelevant parameters in ill-specified formulae will be ignored and the defaults for the parameters will be used instead.
For example, if the model receives the IRF specification ``Normal(alpha=1, beta=1)``, it will initialize a Normal IRF at :math:`\mu=0`, :math:`\sigma=1` (the defaults for this kernel), since :math:`\alpha` and :math:`\beta` are not recognized parameter names for the Normal distribution.
Therefore, make sure to match the parameter names above when specifying parameter defaults.
The correctness of initializations can be checked in the Tensorboard logs.



Using Constant (Non-trainable) Parameters
------------------------------
By default, DTSR trains all the variables that parameterize an IRF kernel (e.g. both :math:`\mu` and :math:`\sigma` for a Gaussian IRF kernel).
But in some cases it's useful to treat certain IRF parameters as constants and leave them untrained.
To do this, specify a list of trainable parameters with the keyword argument ``trainable``, using Python list syntax.
For example, to specify a ShiftedGamma IRF in which the shift parameter :math:`\delta` is held constant at -1, use the following IRF specification::

    ShiftedGamma(delta=-1, trainable=[alpha, beta])

The model will then only train the :math:`\alpha` and :math:`\beta` parameters of the response.
As with parameter initialization, unrecognized parameter names in the ``trainable`` argument will be ignored, and parameter name mismatches can result in more parameters being held constant than intended.
For example, the IRF specification ``Normal(trainable=[alpha, beta])``, will result in an (untrainable) Normal IRF with all parameters held fixed at their defaults.
It is therefore important to make sure that parameter names match those given above.
The correctness of the ``trainable`` specification can be checked in the Tensorboard logs, as well as by the number of trainable parameters reported to standard error at the start of DTSR training.
Constant parameters will show 0 trainable parameters.



Parameter Tying
---------------

A convolutional term in a DTSR model is factored into two components, an IRF component with appropriate parameters and a coefficient governing the overall amplitude of the estimate.
Unless otherwise specified, both of these terms are fit separately for every predictor in the model.
However, parameter tying is possible by passing keyword arguments to the IRF calls in the model formula.
Coefficients can be tied using the ``coef_id`` argument, and IRF parameters can be tied using the ``irf_id`` argument.
For example, the following RHS fits separate IRF and coefficients for each of ``A`` and ``B``::

    C(A, Normal()) + C(B, Normal())

The following fits a single IRF (called "IRF_NAME") but separate coefficients for ``A`` and ``B``::

    C(A, Normal(irf_id=IRF_NAME)) + C(B, Normal(irf_id=IRF_NAME))

The following fits separate IRF but a single coefficient (called "COEF_NAME") for both ``A`` and ``B``::

    C(A, Normal(coef_id=COEF_NAME)) + C(B, Normal(coef_id=COEF_NAME))

And the following fits a single IRF (called "IRF_NAME") and a single coefficient (called "COEF_NAME"), both of which are shared between ``A`` and ``B``::

    C(A, Normal(irf_id=IRF_NAME, coef_id=COEF_NAME)) + C(B, Normal(irf_id=IRF_NAME, coef_id=COEF_NAME))



Transforming Variables
----------------------
DTSR provides limited support for automatic variable transformations based on model formulae.
As in **R** formulae, a transformation is applied by wrapping the predictor name in the transformation function.
For example, to fit a Gamma IRF to a log transform of predictor ``A``, the following is added to the RHS::

    C(log(A), Gamma())

Transformations may be applied to the predictors and/or the response.

The following are the currently supported transformations:

- ``log()``: Applies a natural logarithm transformation to the variable
- ``log1p()``: Adds 1 to the variable an applies a natural logarithm transformation (useful if predictor can include 0)
- ``exp()``: Exponentiates the variable
- ``z()``: Z-transforms the variable (subtracts its mean and divides by its standard deviation)
- ``c()``: 0-centers the variable (subtracts its mean)
- ``s()``: Scales the variable (divides by its standard deviation)

Other transformations must be applied via data preprocessing.



Continuous predictors
---------------------

DTSR's discrete convolution is only exact for discrete impulses (e.g. spikes of stimulus).
Impulse streams that constitute `samples` from a continuous source signal cannot be convolved exactly because the source is generally not analytically integrable.
However, DTSR supports discrete approximation of convolution with continuous inputs through linear interpolation of the impulse between samples, performed at a fixed frequency.

To flag a predictor as continuous, use the ``cont`` keyword argument in the IRF call of the model formula, as shown::

    C(A, Gamma(cont=T))

Be warned that, due to the need for interpolation, continuous predictors tend to impose a heavy computational burden that can dramatically slow training and prediction.
Speedups can be obtained at the expense of accuracy by choose a small value for the **n_interp** initialization parameter, decreasing the resolution of the interpolation.



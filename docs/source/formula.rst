.. _formula:

CDR Model Formulas
===================



Basic Overview
--------------

This package constructs CDR models from **R**-style formula strings defining the model structure.
A CDR formula has the following template::

    RESPONSE ~ FIXED_EFFECTS + RANDOM_EFFECTS

The left-hand side (LHS) of the formula contains the name a single (possibly transformed) variable from the input data table, and the right-hand side (RHS) contains fixed and random effects, each of which must consist exclusively of intercept terms and/or convolutional terms.
Intercept terms can be added by including ``1`` in the RHS and removed by including ``0`` in the RHS.
If neither of these appears in the RHS, an intercept is added by default.

**WARNING:** The compiler for CDR formulas is still in development and may fail to correctly process certain formula strings, especially ones that contain hierarchical term expansions, categorical variable expansions, and/or interactions.
Please double-check the correctness of the model formula reported at the start of training and report problems in the issue tracker on `Github <https://github.com/coryshain/cdr>`_.

**WARNING:** If you are using interaction terms, it is easy to accidentally mis-specify your formula because interactions are trickier in CDR than in linear models.
Make sure to read :ref:`interactions` before fitting models with interactions.

**WARNING:** Column names in your data must be valid Python identifiers in order to be parsed correctly in model formulas.
Numeric names, names containing punctuation other than underscore, and other violations of Python's variable naming rules can result in an incorrect formula parse, which will at best crash and at worse quietly build a different model than the one you wanted.


CDR Model Formulas
------------------

.. _linear:
Linear (DiracDelta) Predictors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A trivial way of "deconvolving" a predictor ``A`` is to assume a Dirac delta response to ``A``: :math:`A` at :math:`t=0`, :math:`0` otherwise.
DiracDelta IRF are thus equivalent to predictors in linear models; only the value of the predictor that coincides with the response measurement is used for prediction.
Linear (mixed) models of time series are therefore a special subtype of CDR, namely CDR models that exclusively use Dirac delta response kernels.
While you therefore `can` use this package to fit LME models, that doesn't mean you should!
The stochastic optimization used here is likely much less efficient in this simple case than that of specialized LME regression tools.

However, the availability of Dirac delta responses in this package means that CDR can combine linear and convolved substructures into a single model, which is important for many use cases.
For example, imagine a bi-variate model with predictors ``stimulus`` and ``condition``, where ``stimulus`` varies within each time series can might engender a temporally diffise response while ``condition`` is held fixed within each time series.
In this case, a convolution of ``condition`` would be meaningless (or at least perfectly confounded with ``rate``, the deconvolutional intercept); instead, we're interested in a `linear` effect of ``condition`` and a `convolved` effect of ``stimulus`` on the response.
Dirac delta responses can specified manually using the syntax described in the following sections.
However, for convenience, bare predictor names in model formulas (outside of convolution calls ``C()``) are automatically interpreted as Dirac delta.
This allows you to retain stock formula syntax that might already be familiar from linear (mixed) models in **R** in order to define linear substructures of your CDR model.
For example, the following is interpreted as specifying linear effects for each of ``A``, ``B``, and the interaction ``A:B``::

    y ~ A + B + C:B

An important caveat in using Dirac delta responses is that the predictors they apply to should be `response-aligned`, that is, measured at the same moments in time that the responses themselves are.
If the stimuli and responses are measured at different points in time, you cannot fit a Dirac delta response to a stimulus-aligned variable unless that variable happens at least sometimes to accidentally be measured concurrently with the response.
CDR does not internally distinguish stimulus-aligned and response-aligned predictors, so it's up to users to make sure that models are sensible in this regard.
However, the model does internally mask all elements in a Dirac delta predictor that are not simultaneous with the response.
Therefore, fitting a Dirac delta response to a stimulus-aligned predictor will not produce an erroneous model; in the worst case, the entire predictor will be masked and therefore ignored by the model.

Defining an Impulse Response Function (IRF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A convolutional term is defined using the syntax ``C(..., IRF_FAMILY())``, where ``...`` is replaced by names of predictors contained in the input data.
For example, to define a Gamma convolution of predictor ``A``, the expression ``C(A, Gamma())`` is added to the RHS.
Separate terms are delimited by ``+``.
For example, to add a Gaussian convolution of predictor ``B``, the RHS above becomes ``C(A, Gamma()) + C(B, Normal())``.


Supported Parametric IRFs
^^^^^^^^^^^^^^^^^^^^^^^^^

The currently supported parametric IRF families are the following.
All IRFs are normalized to integrate to 1 over the positive real line.
For simplicity, normalization constants are omitted from the equations below.
For details, see Shain & Schuler (2021).

- ``DiracDelta``: Stick function (equivalent to a predictor in linear regression, see :ref:`linear`)

  - Parameters:

    - None

  - Definition: :math:`1` at :math:`x=0`, :math:`0` otherwise

- ``Exp``: PDF of exponential distribution

  - Parameters:

    - :math:`\beta > 0`: ``beta`` (rate)

  - Definition: :math:`\beta e^{-\beta x}`
  - Note: Dropping the normalization constant from the PDF helps deconfound IRF shape and magnitude. Normalization is unnecessary since this kernel defines an IRF, not a probability distribution.

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
    - :math:`\sigma^2 > 0`: ``sigma2`` (variance)

  - Definition: :math:`\frac{1}{\sqrt{2\pi \sigma^2}}e^{-\frac{(x - \mu) ^ 2}{2 \sigma ^ 2}}`
  - Note: Dropping the normalization constant from the PDF helps deconfound IRF shape and magnitude. Normalization is unnecessary since this kernel defines an IRF, not a probability distribution.

- ``SkewNormal``: PDF of SkewNormal distribution (normal distribution augmented with left/right skew parameter)

  - Parameters:

    - :math:`\mu` (mean)
    - :math:`\sigma > 0` (standard deviation)
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

- ``HRFDoubleGamma1``: 1-parameter double-gamma hemodynamic response function (fMRI). Shape parameters are fixed at SPM's defaults for both the first and second gammas (6 and 16, respectively). Parameter :math:`\beta` is tied between both gammas. The coefficient on the second gamma is fixed at SPM's default (:math:`\frac{1}{6}`). This is a "stretchable" canonical HRF.

  - Parameters:

    - :math:`\beta > 0`: ``beta`` (peak and undershoot rate)

  - Definition: :math:`\frac{\beta^{6}x^{6-1}e^{-\frac{x}{\beta}}}{\Gamma(6)} - \frac{1}{6}\frac{\beta^{16}x^{15}e^{-\frac{x}{\beta}}}{\Gamma(16)}`

- ``HRFDoubleGamma2``: 2-parameter double-gamma hemodynamic response function (fMRI). Parameter :math:`\alpha` of the second gamma is fixed to the :math:`alpha` of the first gamma using SPM
s default offset (10). Parameter :math:`\beta` is tied between both gammas. The coefficient on the second gamma is fixed at SPM's default (:math:`\frac{1}{6}`).

  - Parameters:

    - :math:`\alpha > 1`: ``alpha`` (peak shape)
    - :math:`\beta > 0`: ``beta`` (peak and undershoot rate)

  - Definition: :math:`\frac{\beta^{\alpha}x^{\alpha-1}e^{-\frac{x}{\beta}}}{\Gamma(\alpha)} - \frac{1}{6}\frac{\beta^{\alpha + 10}x^{\alpha + 9}e^{-\frac{x}{\beta}}}{\Gamma(\alpha + 10)}`

- ``HRFDoubleGamma3``: 3-parameter double-gamma hemodynamic response function (fMRI). Parameter :math:`\alpha` of the second gamma is fixed to the :math:`alpha` of the first gamma using SPM
s default offset (10). Parameter :math:`\beta` is tied between both gammas.

  - Parameters:

    - :math:`\alpha > 1`: ``alpha`` (peak shape)
    - :math:`\beta > 0`: ``beta`` (peak and undershoot rate)
    - :math:`c`: ``c`` (undershoot coefficient)

  - Definition: :math:`\frac{\beta^{\alpha}x^{\alpha-1}e^{-\frac{x}{\beta}}}{\Gamma(\alpha)} - c\frac{\beta^{\alpha + 10}x^{\alpha + 9}e^{-\frac{x}{\beta}}}{\Gamma(\alpha + 10)}`

- ``HRFDoubleGamma4``: 4-parameter double-gamma hemodynamic response function (fMRI). Parameter :math:`\beta` is tied between both gammas.

  - Parameters:

    - :math:`\alpha_1 > 1`: ``alpha_main`` (peak shape)
    - :math:`\alpha_2 > 1`: ``alpha_undershoot`` (undershoot shape)
    - :math:`\beta > 0`: ``beta`` (peak and undershoot rate)
    - :math:`c`: ``c`` (undershoot coefficient)

  - Definition: :math:`\frac{\beta^{\alpha_1}x^{\alpha_1-1}e^{-\frac{x}{\beta}}}{\Gamma(\alpha_1)} - c\frac{\beta^{\alpha_2}x^{\alpha_2 - 1}e^{-\frac{x}{\beta}}}{\Gamma(\alpha_2)}`

- ``HRFDoubleGamma5``: 5-parameter double-gamma hemodynamic response function (fMRI). All parameters are free.

  - Parameters:

    - :math:`\alpha_1 > 1`: ``alpha_main`` (peak shape)
    - :math:`\alpha_2 > 1`: ``alpha_undershoot`` (undershoot shape)
    - :math:`\beta_1 > 0`: ``beta_main`` (peak rate)
    - :math:`\beta_2 > 0`: ``beta_undershoot`` (undershoot rate)
    - :math:`c`: ``c`` (undershoot coefficient)

  - Definition: :math:`\frac{\beta^{\alpha_1}x^{\alpha_1-1}e^{-\frac{x}{\beta_1}}}{\Gamma(\alpha_1)} - c\frac{\beta^{\alpha_2}x^{\alpha_2 - 1}e^{-\frac{x}{\beta_2}}}{\Gamma(\alpha_2)}`


.. _interactions:

Interactions in CDR
^^^^^^^^^^^^^^^^^^^

In comparison to interactions in linear models, deconvolution introduces the additional complexity of needing to decide and specify whether interactions precede (impulse-level interactions) or follow (response-level interactions) the convolution step.
Impulse-level interactions consider interactions as `events` which may trigger a temporally diffuse response (i.e. a response to both A and B happening together at a particular point in time).
Response-level interactions capture non-additive effects of multiple (possibly convolved) variables; they do not get their own impulse responses.
Response-level interactions correspond to interactions in linear models and are almost always what you want except in the special case of linear (DiracDelta IRF) predictors, where impulse-level interactions should be used (just like in linear models).

CDR formulas use a simple syntax to distinguish these two types of interactions: impulse-level interactions are specified `inside` the first argument of convolution calls `C()`, while response-level interactions are specified outside them.
As in **R**, interaction terms are designated with ``:``, as in ``A:B``.
And as in **R**, for convenience, two-way cross-product interactions can be designated with ``*`` (e.g. ``A*B`` is shorthand for ``A + B + A:B``) and multi-way cross-product interactions can be designated with power notation ``^<INT>`` or ``**<INT>`` (e.g. ``(A+B+C)^3`` equals ``A + B + C + A:B + B:C + A:C + A:B:C``).
The following defines an impulse-level interaction between ``A`` and ``B`` underneath a ``Normal`` IRF kernel::

    C(A:B, Normal())

The following defines a response-level interaction between Normal convolutions of ``A`` and ``B``::

    C(A, Normal()):C(B, Normal())

In order to fit interactions between convolved variables, the convolutions themselves must exist.
Therefore, unlike linear interactions, which can be fit even if their subcomponents are not included in the model, ``C(A, Normal()):C(B, Normal())`` requires the existence of model estimates for both ``C(A, Normal())`` and ``C(B, Normal())``, and these terms are therefore automatically inserted when used by any response-level interactions.

Response-level interactions do not need to be convolved variables.
They can also be predictors supplied by the data `as long as the predictors are response-aligned` (i.e. measured concurrently with the responses, rather than the impulses).
For example, suppose we have a response-aligned variable ``C`` provided by our data.
We can interact responses with it, like so::

    C(A, Normal()):C

This will fit a normal response to A, along with an estimate for the modulation of that response by C.
Unlike convolved inputs to response-level interactions, estimates for regular variables are not automatically added to the model.
In order to fit a separate (linear) effect for C, we could use the multiplication operator instead::

    C(A, Normal())*C = C(A, Normal() + C + C(A, Normal()):C

For convenience, response-level interactions distribute across the inputs to a convolution call ``C()``.
Thus, interacting a variable with a convolution of multiple inputs is equivalent to interacting the variable with a convolution of each of the inputs::

    C(A + B, Gamma()):C = C(A + B, Gamma()) + C(A, Gamma()):C + C(B, Gamma()):C

Similarly, interacting multiple convolution calls each containing multiple inputs is equivalent to defining interactions over the Cartesian-product of the responses to the two sets of inputs::

    C(A + B, Gamma()):C(C + D, EMG()) = C(A + B, Gamma()) + C(C + D, EMG()) + \
                                        C(A, Gamma()):C(C, EMG()) + C(B, Gamma()):C(C, EMG()) + \
                                        C(A, Gamma()):C(D, EMG()) + C(B, Gamma()):C(D, EMG())

Order of operations between term expansions can be enforced through parentheses::

    (A*B):E = A:E + B:E + A:B:E
    A*(B:E) = A + B:E + A:B:E



Automatic Term Expansion
^^^^^^^^^^^^^^^^^^^^^^^^

For convenience, the ``C()`` function distributes the impulse response family over multiple ``+``-delimited terms in its first argument.
Therefore, the following two expressions are equivalent::

    C(A + B, Gamma())
    C(A, Gamma()) + C(B, Gamma())



**R**-style expansions for interactions are also available, as discussed above.
IRF distribute across the expansion of interaction terms, such that the following expressions are equivalent::

    C((A + B + C)**3, Gamma())
    C(A, Gamma()) + C(B, Gamma()) + C(C, Gamma()) + C(A:B, Gamma()) + C(B:C, Gamma()) + C(A:C, Gamma()) + C(A:B:C, Gamma())

Categorical variables are automatically discovered and expanded in CDR models.
This process imposes a transformation on the model.
For example, imagine that predictor ``B`` in the following model turns out to be categorical in the data set with categories ``B1``, ``B2``, and ``B3``::

    C(A + B, EMG())

When the CDR model is initialized, the categorical nature of ``B`` is detected and the model is expanded out as::

    C(A + B2 + B3, EMG())


However, they can be included simply by adding binary indicator vectors for each of :math:`n-1` of the levels of the variable to the input data as a preprocessing step, then defining the model in terms of the binary indicators.

Note that the term expansions described above add `separate` IRF for each term in the expansion.
For example, ``C(A + B, Gamma())`` adds two distinct Gamma IRF parameterizations to the model, one for each predictor.
It is also possible to tie IRF between predictor variables (details below).

Note also that (unlike **R**) redundant terms are **not** automatically collapsed, so care must be taken to ensure that no duplicate terms are produced via term expansion.



Random Effects
^^^^^^^^^^^^^^

Random effects in CDR are specified using the following syntax::

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

An important complication in fitting mixed models with CDR is that the relevant grouping factor is determined by the current `regression target`, not the properties of the independent variable observations in the series history.
This means that random effects are only guaranteed to be meaningful when fit using grouping factors that are constant for the entire series (e.g. the ID of the human subject completing the experiment).
Random effects fit for grouping factors that vary during the experiment should therefore be avoided unless they are intercept terms only, which are not affected by the temporal convolution.



Parameter Initialization
^^^^^^^^^^^^^^^^^^^^^^^^

IRF parameters can be initialized for a given convolutional term by specifying their initial values in the IRF call, using the parameter name as the keyword (see supported IRF and their associated parameters above).
For example, to initialize a Gamma IRF with :math:`\alpha = 2` and :math:`\beta = 2` for predictor ``A``, use the following call::

    C(A, Gamma(alpha=2, beta=2))

These values will serve as initializations in both CDRMLE and CDRBayes, and in CDRBayes they will additionally serve as the mean of the prior distribution for that parameter.
If no initialization is specified, defaults will be used.
These defaults are not guaranteed to be plausible for your particular application and may have a detrimental impact on training.
Therefore, it is generally a good idea to think carefully in advance about what kinds of IRF shapes are `a priori` reasonable and choose initializations in that range.

Note that the initialization values are on the constrained space, so make sure to respect the constraints when choosing them.
For example, :math:`\alpha` of the Gamma distribution is constrained to be > 0, so an initial :math:`\alpha` of <=0 will result in incorrect behavior.
However, keep in mind that for CDRBayes, prior variances are necessarily on the unconstrained space and get squashed by the constraint function, so choosing initializations that are very close to constraint boundaries can indirectly tighten the prior.
For example, choosing an initialization :math:`\alpha = 0.001` for the Gamma distribution will result in a much tighter prior around small values of :math:`\alpha`.

Initializations for irrelevant parameters in ill-specified formulas will be ignored and the defaults for the parameters will be used instead.
For example, if the model receives the IRF specification ``Normal(alpha=1, beta=1)``, it will initialize a Normal IRF at :math:`\mu=0`, :math:`\sigma=1` (the defaults for this kernel), since :math:`\alpha` and :math:`\beta` are not recognized parameter names for the Normal distribution.
Therefore, make sure to match the parameter names above when specifying parameter defaults.
The correctness of initializations can be checked in the Tensorboard logs.



Using Constant (Non-trainable) Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, CDR trains all the variables that parameterize an IRF kernel (e.g. both :math:`\mu` and :math:`\sigma` for a Gaussian IRF kernel).
But in some cases it's useful to treat certain IRF parameters as constants and leave them untrained.
To do this, specify a list of trainable parameters with the keyword argument ``trainable``, using Python list syntax.
For example, to specify a ShiftedGamma IRF in which the shift parameter :math:`\delta` is held constant at -1, use the following IRF specification::

    ShiftedGamma(delta=-1, trainable=[alpha, beta])

The model will then only train the :math:`\alpha` and :math:`\beta` parameters of the response.
As with parameter initialization, unrecognized parameter names in the ``trainable`` argument will be ignored, and parameter name mismatches can result in more parameters being held constant than intended.
For example, the IRF specification ``Normal(trainable=[alpha, beta])``, will result in an (untrainable) Normal IRF with all parameters held fixed at their defaults.
It is therefore important to make sure that parameter names match those given above.
The correctness of the ``trainable`` specification can be checked in the Tensorboard logs, as well as by the number of trainable parameters reported to standard error at the start of CDR training.
Constant parameters will show 0 trainable parameters.



Parameter Tying
^^^^^^^^^^^^^^^

A convolutional term in a CDR model is factored into two components, an IRF component with appropriate parameters and a coefficient governing the overall amplitude of the estimate.
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
^^^^^^^^^^^^^^^^^^^^^^

CDR provides limited support for automatic variable transformations based on model formulas.
As in **R** formulas, a transformation is applied by wrapping the predictor name in the transformation function.
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



Pseudo Non-Parametric IRFs
^^^^^^^^^^^^^^^^^^^^^^^^^^

CDR also supports pseudo non-parametric IRFs in the form of Gaussian kernel functions (linear combination of Gaussians or LCG).
Instead of a parametric IRF kernel, the model implements the IRF as a sum of Gaussian kernel functions whose location, spread, and height can be optimized by the model.
The advantage of LCG IRFs is that they do not require precommitment to a particular functional form for the IRF.
The disadvantage is that fitting them is slower because they involve more parameters and computation.

The kernels themselves have a number of free parameters which are specified by the name of the kernel in the IRF call of the model formula.
The syntax for an LCG IRF kernel is as follows::

    LCG(b([0-9]+))?

This is a string representation of a function call ``LCG`` with optional keyword argument ``b``.

The keyword argument is defined as follows:

  - **b** (bases): ``int``, number of bases (control points). **Default**: 10.



IRF Composition
^^^^^^^^^^^^^^^

In some cases it may be desirable to decompose the response into multiple convolutions of an impulse.
For example, it is possible that the BOLD response in fMRI consists underlyingly of 2 convolutional responses: a **neural response** that convolves the impulse into a timecourse of neural activation, which is then convolved with a **hemodynamic response** into a BOLD signal.
In this case, it would be desirable to be able to model the BOLD response as a composition of neural and hemodynamic responses.

Exact parametric composition of IRF is not possible in the general case because many pairs of IRF do not have a tractable analytical convolution.
Instead, the CDR package uses a discrete approximation to the continuous integral of composed IRF by (1) computing the value of each IRF for some number of interpolation points, (2) computing their convolution via FFT, and (3) rescaling by the temporal distance between interpolation points.
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

CDR is not able to recognize and flag identifiability problems and it will happily find a solution to such a model, disguising the fact that there are infinitely many other optimal solutions.
It is up to the user to think carefully about whether the model structure could introduce such problems.
For example, in the BOLD example discussed above, the neural response is predictor-specific while the hemodynamic response is predictor-independent given the neural response.
The two responses can thus be separated via parameter tying of the hemodynamic response portion (see below), requiring all predictors to share a single hemodynamic response and forcing predictor-level variation into the neural response alone.
**NOTE:**: Only parametric (not neural network) IRFs can be composed in this way. Numerical integration of neural network IRFs is computationally prohibitive.


Neural Network Components
-------------------------

CDR allows two kinds of neural network components in the model architecture.
First, rather than using a parametric IRF kernel, you can use a deep neural network IRF, simply by using the term ``NN()`` as the second argument of a ``C()`` call::

    y ~ C(A + B, NN())

NN hyperparameters can either be globally defined through keywords in ``[cdr_settings]`` or locally defined in the formula via keyword arguments to ``NN()`` (for available options, see :ref:`config`).
The main reason to define a hyperparameter locally within the formula is if you want to override a global setting for a particular neural network component.
For example, imagine we have two predictors ``A`` and ``B`` and we want to constrain the response to be linear on ``A`` (but not ``B``).
We can achieve this by varying the value of the ``input_dependent_irf`` setting (which determines whether the IRF is allowed to differ at different input values, opening the possibility of non-linear responses), as follows::

    [model_CDR_example]
    input_dependent_irf = True
    formula = y ~ C(A, NN(input_dependent_irf=False)) + C(B, NN())

In the above, the default setting for ``input_dependent_irf`` is set to ``True`` in the model settings (thus, non-linear by default), but this default is overridden in the NN response to ``A`` by using the keyword argument ``input_dependent_irf=False`` in the relevant ``NN()`` call of the model formula.
Using this definition, the IRF to ``B`` depends on the value of ``B``, but the IRF to ``A`` is independent of the value of ``A``, and thus linear.

Just like other IRFs, neural network IRFs can participate in both fixed and random effects.
For example, the following defines a single population-level neural network IRF but allows the coefficients to vary by subject::

    y ~ C(A + B, NN()) + (C(A + B, NN()) | subject)

By contrast, the following allows by-subject variation in both in the coefficients and in the neural network parameters themselves::

    y ~ C(A + B, NN()) + (C(A + B, NN(ran=T)) | subject)

In addition, formulas can include trainable neural network transformations of predictors, simply by placing ``NN()`` in the first argument of a ``C()`` call and entering the sum of predictors to transform as its argument::

    y ~ C(NN(A + B + C), Normal())

The above formula will first apply a feedforward neural network to transform the vector ``[A, B, C]`` into a scalar, which is then treated as a predictor with a ``Normal`` IRF kernel.
Neural net predictor transforms can also take neural network IRFs::

    y ~ C(NN(A + B + C), NN())

In this case, one NN defines a transform on ``A``, ``B``, and ``C``, and another defines an IRF that describes the diffusion of the effect of that transformed value over time.
Note that in the above formula, the two NNs operate independently, one transforming data, the other convolving it.
But this implementation also supports input-dependent NN IRFs that take the input values into account in determining the IRF shape.
Input-dependent IRFs can be used by setting the ``input_dependent_irf`` field to ``True`` in the ``[cdr_settings]`` section of the config.
Input-dependent IRFs make predictor transformations unnecessary (since the IRF implicitly represents a transformation of the predictors), so the above formula would only make sense if input-dependence were turned off.

Parametric IRFs distribute over their inputs. Thus, the following two formulas are equivalent, and both express distinct IRF transforms for the variables A and B::

    y ~ C(A + B, Normal())
    y ~ C(A, Normal()) + C(B, Normal())

However, this is not the case for deep neural IRFs, where the elements in the first argument of ``C()`` determine the number of convolution weights that the neural IRF will generate.
Thus, the following two formulas are not equivalent::

    y ~ C(A + B, NN())
    y ~ C(A, NN()) + C(B, NN())

The first defines a single NN IRF with two outputs that jointly convolves ``A`` and ``B``.
The second defines two distinct NN IRFs, each with one output, that separately convolve ``A`` and ``B``.

Neural components can be flexibly combined with non-neural components.
For example, the following treats ``A`` as a linear (``DiracDelta``) predictor, convolves ``B`` with a neural IRF, and convolves ``C`` with a ``Normal`` IRF::

    y ~ A + C(B, NN()) + C(C, Normal())



Multivariate Responses
----------------------

CDR can jointly model multiple response variables.
Unless the model contains neural components, this is equivalent to fitting a distinct CDR model to each response vector, but it can be more computationally efficient.
When neural components are used, distributed representations entail that multivariate models can substantively differ from separate univariate models.
Information about one response variable can inform inferences made about other response variables.

To model multiple response variables, simply enter them all on the left-hand side of the model formula, delimited by ``+``.
For example, the following jointly models ``y1`` and ``y2`` as a function of ``A`` and ``B``::

    y1 + y2 ~ C(A + B, Gamma())



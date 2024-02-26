# Continuous-Time Deconvolutional Regression (CDR)

In many real world time series, events trigger "ripples" in a dependent variable that unfold slowly and overlap in time (temporal diffusion).
Recovering the underlying dynamics of temporally diffuse effects is challenging when events and/or responses occur at irregular intervals.
Continuous-time deconvolutional regression (CDR) is a regression technique for time series that directly models temporal diffusion of effects (Shain & Schuler, 2018, 2021) as a funtion of continuous time.
CDR uses machine learning to estimate continuous-time impulse response functions (IRFs) that mediate between predictors (event properties) and responses.
Given data and a model template specifying the functional form(s) of the IRF kernel(s), CDR finds IRF parameters that optimize some objective function.
CDR is a form of distributional regression: it can model continuous-time influences of predictors on all parameters of arbitrary predictive distributions, not just on the mean of a normal predictive distribution.
This approach can be generalized to account for non-stationarity, non-linearity, non-additivity, and context-dependence by incorporating deep neural network components (Shain, 2021).

This repository contains source code for the `cdr` Python module as well as support for reproducing published experiments (see [Reproducing published results](#reproducing-published-results) below).
This package provides (1) an API for programming with CDR and (2) executables that allow users to train and evaluate CDR models out of the box, without needing to write any code.
Full documentation for the `cdr` module is available at [http://cdr.readthedocs.io/en/latest/](http://cdr.readthedocs.io/en/latest/).
This repository also provides an experimental tool for interactive browser-based visualization of CDR estimates.

CDR models can be trained and evaluated using provided utility executables.
Help strings for all available utilities can be viewed by running `python -m cdr.bin.help`.
Full repository documentation, including an API, is provided at the link above.


## Installation

### PyPI

Install [python](https://www.python.org/), then install CDR(NN) using ``pip`` as follows:

    pip install cdrnn

Note that, despite the package name, both (kernel-based) CDR and (deep-neural) CDRNN are implemented by the same codebase, so both kinds of models can be fitted using this package.

### Anaconda

Install [anaconda](https://www.anaconda.com/), then run the following commands from this repository root to create a new conda environment:

    conda env create -f conda_cdr.yml
    conda activate cdr
    python setup.py install
    
The `cdr` environment must first be activated anytime you want to use the CDR codebase:

    conda activate cdr


## Basic usage

Once the `cdr` package is installed system-wide as described above (and the `cdr` conda environment is activated via `conda activate cdr`), the `cdr` package can be imported into Python as shown

    import cdr
    
Most users will not need to program with CDR(NN), but will instead run command-line executables for model fitting and criticism.
These can be run as

    python -m cdr.bin.<EXECUTABLE-NAME> ...
    
For documentation of available CDR(NN) executables, run

    python -m cdr.bin.help( <SCRIPT-NAME>)*

CDR(NN) models are defined using configuration (`*.ini`) files, which can be more convenient than shell arguments for specifying many settings at once, and which provide written documentation of the specific settings used to generate any given result.
For convenience, we have provided a utility to initialize a new `*.ini` file, which can be run like this:

    python -m cdr.bin.create_config > PATH.ini
    
This will initialize a CDR-oriented config file.
To initialize a CDRNN-oriented config file, add the flag `-t cdrnn`.
To initialize a plotting-oriented config file (which defines visualizations to run for an already fitted model), add the flag `-t plot`.
To include annotation comments in the output file (which can help with customizing it), add the flag `-a`.
The `PATH.ini` file can then be modified as needed to set up the analysis.

CDR model formula syntax resembles R-style model formulas (`DV ~ IV1 + IV2 + ...`) and is fully described in the [docs](http://cdr.readthedocs.io/en/latest/).
The core novelty is the `C(preds, IRF)` call (`C` for "convolve"), in which the first argument is a '+'-delimited list of predictors and the second argument is a call to an impulse response kernel (e.g. `Exp`, `Normal`, `ShiftedGammaShapeGT1`, see docs for complete list).
For example, a model with the following specification

    formula = DV ~ C(a + b + c, NN())
    
will fit a CDRNN model that convolves predictors `a`, `b`, `c` using a deep neural impulse response function (IRF).

Once a model file has been written (e.g. `model.ini`), the model(s) defined in it can be trained by running:

    python -m cdr.bin.train model.ini
    
IRF estimates will be incrementally dumped into the output directory specified by `model.ini`,
and learning curves can be inspected in Tensorboard:

    python -m tensorboard.main --logdir=<PATH-TO-CDR-OUTPUT>

For more on usage, see the [docs](http://cdr.readthedocs.io/en/latest/).


## Interactive visualization

We are in the process of developing a web-based tool for interactive visualization of CDR estimates.
This is especially useful in models with neural network components, where estimates can contain rich interactions.
To play with the current version of the tool, run:

    python -m cdr.bin.viz <PATH-TO-CDR-MODEL-FILE> -m <MODEL-NAME>
    
and open the URL displayed in the console in a web browser.
You can then use the provided interface to explore many aspects of your model's estimates.
This tool is still in very early stages of development, and feedback is welcome.


## Reproducing published results

This repository is under active development, and reproducibility of previously published results is not guaranteed from the master branch.
For this reason, repository states associated with previous results are saved in Git branches.
To reproduce those results, checkout the relevant branch and follow the instructions in the `README`.
Current reproduction branches are:

 - `emnlp18` (Shain & Schuler, 2018)
 - `naacl19` (Shain, 2019)
 - `npsy` (Shain, Blank, et al., 2020)
 - `cognition21` (Shain & Schuler, 2021)
 - `acl21` (Shain, 2021)
 - `fMRI_ns_WM` (Shain et al., 2022)
 - `cdrnn_journal` (Shain & Schuler, 2024)
 - `pred_fn` (Shain et al., 2024)
 - `freq_pred` (Shain, 2024)

Thus, to reproduce results from ACL21, for example, run `git checkout acl21` from the repository root, and follow instructions in the `README` file.
The reproduction branches are also useful sources of example configuration files to use as templates for setting up your own experiments, although you should consult the docs for full documentation of the structure of CDR experiment configurations.

Published results depend on both (1) datasets and (2) models as defined in experiment-specific configuration files.
In general, we do not distribute data with this repository.
The datasets used can be provided by email upon request.

Note that some published experiments also involve fitting LME and GAM models, which require `rpy2` and therefore won't work on Windows systems without some serious hacking.
The `cdr` module is cross-platform and therefore CDR models should train regardless of operating system.


## Help and support

For questions, concerns, or data requests, contact Cory Shain ([cory.shain@gmail.com](cory.shain@gmail.com)).
Bug reports can be logged in the issue tracker on [Github](https://github.com/coryshain/cdr).


## References
Shain, Cory and Schuler, William (2018). Deconvolutional time series regression: A technique for modeling temporally diffuse effects. _EMNLP18_.

Shain, Cory (2019). A large-scale study of the effects of word frequency and predictability in naturalistic reading. _NAACL19_.

Shain, Cory and Blank, Idan and van Schijndel, Marten and Schuler, William and Fedorenko, Evelina (2020). fMRI reveals language-specific predictive coding during naturalistic sentence comprehension. _Neuropsychologia_.

Shain, Cory and Schuler, William (2021). Continuous-time deconvolutional regression for psycholinguistic modeling. _Cognition_.

Shain, Cory (2021). CDRNN: Discovering complex dynamics in human language processing. _ACL21_.

Shain, Cory and Blank, Idan and Fedorenko, Evelina and Gibson, Edward and Schuler, William (2022). Robust effects of working memory demand during naturalistic language comprehension in language-selective cortex. _Journal of Neuroscience_.

Shain, Cory and Schuler, William (2024). A deep learning approach to analyzing continuous-time systems. _Open Mind_.

Shain, Cory and Meister, Clara and Pimentel, Tiago and Cotterell, Ryan and Levy, Roger (2024). Large-Scale Evidence for Logarithmic Effects of Word Predictability on Reading Time. _PNAS_.

Shain, Cory (2024). Word frequency and predictability dissociate in naturalistic reading. _Open Mind_.

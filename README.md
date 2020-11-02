# Continuous-Time Deconvolutional Regression (CDR)
CDR (formerly _deconvolutional time series regression_ or _DTSR_) is a regression technique for analyzing time series with delayed effects (Shain & Schuler, 2018, 2019).

This repository contains source code for the `cdr` Python module as well as support for reproducing published experiments.
Full documentation for the `cdr` module is available at [http://dtsr.readthedocs.io/en/latest/](http://dtsr.readthedocs.io/en/latest/).

CDR models can be trained and evaluated using provided utility executables.
Help strings for all available utilities can be viewed by running `python -m cdr.bin.help`.
Full repository documentation, including an API, is provided at the link above.

This repository also provides alpha support for CDRNN (*manuscript in preparation*), a deep neural extension of CDR that models the impulse response as a non-stationary, context-dependent manifold over the predictor and time dimensions.
CDRNN eliminates many simplifying assumptions made by CDR and improves its ability to capture non-linear featural and temporal relationships between the predictors and the response.

## Installation

Unfortunately, because the TensorFlow API evolves quickly and support has been discontinued for the Edward library, two different installations are needed depending on whether you want to run variational Bayesian CDR on the one hand or CDRNN on the other.
We are currently in the process of trying to upgrade our variational implementation to use Tensorflow Probability instead, but meanwhile, we strongly encourage the use of conda environments to manage installations.
Installing into a [conda](https://www.anaconda.com/) environment avoids version clashes between CDR package requirements and your local software environment.
Once you have installed Anaconda, use the relevant set of instructions below

### CDR Installation

Run the following commands from this repository root to install (variational Bayesian) CDR-oriented dependencies:

    conda env create -f conda_cdr.yml
    conda activate cdr
    python setup.py install
    
Under this approach, the `cdr` environment must first be activated anytime you want to use the CDR codebase:

    conda activate cdr

### CDRNN Installation

Run the following commands from this repository root to install CDRNN-oriented dependencies:

    conda env create -f conda_cdrnn.yml
    conda activate cdrnn
    python setup.py install
    
Under this approach, the `cdrnn` environment must first be activated anytime you want to use the CDR codebase:

    conda activate cdrnn


## Basic usage

Once CDR is installed system-wide as described above (and the `cdr` conda environment is activated via `conda activate cdr` or `conda activate cdrnn`, see above), the CDR package can be imported into Python as shown

    import cdr
    
Most users will not need to program with CDR, but will instead run command-line executables for model fitting and criticism.
These can be run as

    python -m cdr.bin.<EXECUTABLE-NAME> ...
    
For documentation of available CDR executables, run

    python -m cdr.bin.help( <SCRIPT-NAME>)*

CDR models are defined using configuration (`*.ini`) files, which can be more convenient than shell arguments for specifying many settings at once, and which provide written documentation of the specific settings used to generate any given result.
For convenience, we have provided annotated CDR and CDRNN model templates:
    
    cdr_model_template.ini
    cdrnn_model_template.ini
    
These files can be duplicated and modified (e.g. with paths to data and model specifications) in order to quickly get a CDR model up and running.

CDR model formula syntax resembles R-style model formulas (`DV ~ IV1 + IV2 + ...`) and is fully described in the [docs](http://dtsr.readthedocs.io/en/latest/).
The core novelty is the `C(preds, IRF)` call (`C` for "convolve"), in which the first argument is a '+'-delimited list of predictors and the second argument is a call to an impulse response kernel (e.g. `Exp`, `Normal`, `ShiftedGammaShapeGT1`, see docs for complete list).
For example, a modle with the following specification

    formula = DV ~ C(a + b + c, Normal())
    
will fit a CDR model that convolves predictors `a`, `b`, `c` using `Normal` IRFs with trainable location and scale parameters.

Once a model file has been written (e.g. `model.ini`), the model(s) defined in it can be trained by running:

    python -m cdr.bin.train model.ini
    
IRF estimates will be incrementally dumped into the output directory specified by `model.ini`,
and learning curves can be inspected in Tensorboard:

    python -m tensorboard.main --logdir=<PATH-TO-CDR-OUTPUT>

For more on usage, see the [docs](http://dtsr.readthedocs.io/en/latest/).


## Reproducing published results

This repository is under active development, and reproducibility of previously published results is not guaranteed from the master branch.
For this reason, repository states associated with previous results are saved in Git branches.
To reproduce those results, checkout the relevant branch and follow the instructions in the `README`.
Current reproduction branches are:

 - `emnlp18`
 - `naacl19`

Thus, to reproduce results from NAACL19, for example, run `git checkout naacl19` from the repository root, and follow instructions in the `README` file.
The reproduction branches are also useful sources of example configuration files to use as templates for setting up your own experiments, although you should consult the docs for full documentation of the structure of CDR experiment configurations.

Published results depend on both (1) datasets and (2) models as defined in experiment-specific configuration files.
In general, we do not distribute data with this repository.
The datasets used can be provided by email upon request.

Note that some published experiments below also involve fitting LME and GAM models, which require `rpy2` and therefore won't work on Windows systems without some serious hacking.
The `cdr` module is cross-platform and therefore CDR models should train regardless of operating system.

## Help and support

For questions, concerns, or data requests, contact Cory Shain ([shain.3@osu.edu](shain.3@osu.edu)).
Bug reports can be logged in the issue tracker on [Github](https://github.com/coryshain/dtsr).


## References
Shain, Cory and Schuler, William (2018). Deconvolutional time series regression: A technique for modeling temporally diffuse effects. _EMNLP18_.
Shain, Cory and Schuler, William (2019). Continuous-time deconvolutional regression for psycholinguistic modeling. _PsyArXiv_. [https://doi.org/10.31234/osf.io/whvk5](https://doi.org/10.31234/osf.io/whvk5).
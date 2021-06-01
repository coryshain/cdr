# Continuous-Time Deconvolutional Regression (CDR)
CDR (formerly _deconvolutional time series regression_ or _DTSR_) is a regression technique for analyzing time series with delayed effects (Shain & Schuler, 2018, 2021).
Continuous-time deconvolutional regressive neural networks (CDRNNs) are a deep neural generalization of the CDR technique (Shain, 2021).

This repository contains source code for the `cdr` Python module as well as support for reproducing published experiments.
Full documentation for the `cdr` module is available at [http://dtsr.readthedocs.io/en/latest/](http://dtsr.readthedocs.io/en/latest/).

CDR models can be trained and evaluated using provided utility executables.
Help strings for all available utilities can be viewed by running `python -m cdr.bin.help`.
Full repository documentation, including an API, is provided at the link above.

## Installation

Install [anaconda](https://www.anaconda.com/), then run the following commands from this repository root to create a new conda environment:

    conda env create -f conda_cdr.yml
    conda activate cdr
    python setup.py install
    
The `cdr` environment must first be activated anytime you want to use the CDR codebase:

    conda activate cdr

## Basic usage

Once CDR is installed system-wide as described above (and the `cdr` conda environment is activated via `conda activate cdr`), the CDR package can be imported into Python as shown

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

CDRNN models do not require user-specification of a kernel family, so their formula syntax is simpler:

    formula = DV ~ a + b + c

The response shape to all variables (and all their interactions) will be fitted jointly.

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
 - `npsy`
 - `cognition21`
 - `acl21`

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

Shain, Cory and Schuler, William (2021). Continuous-time deconvolutional regression for psycholinguistic modeling. _Cognition_.

Shain, Cory (2021). CDRNN: Discovering complex dynamics in human language processing. _ACL21_.

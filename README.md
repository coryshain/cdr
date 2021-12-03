# Continuous-Time Deconvolutional Regression (CDR)
CDR is a regression technique for modeling temporally diffuse effects (Shain & Schuler, 2018, 2021).

This branch (`cognition21`) exists to support reprodiction of results reported in Shain & Schuler (2021).
It locks the repository at a previous state and therefore lacks any subsequent improvements or bug fixes.
Do not use this branch to run regressions on your own data.
Instead, first run the following command from the repository root:

`git checkout -b master`

Installation of dependencies can be managed through Anaconda (https://www.anaconda.com/).
Once you have an Anaconda distribution installed, the software environment can be set up by running the following from the root of this repository:

`conda env create -f cognition.yml`

Once complete, activate the conda environment as follows:

`conda activate cognition`

## _Cognition_ (2021) results

Published results depend on both (1) datasets and (2) models as defined in experiment-specific configuration files.
The synthetic, self-paced reading, and fMRI data are available on OSF: https://osf.io/hb5w2/.
The eye-tracking (Dundee) data are not publicly available but can be provided by email upon request (cory.shain@gmail.com).

Preprocessing for the fMRI baselines was performed by passing the language network predictor and response dataframes from Shain, Blank, et al. (2020) (available at https://osf.io/eyp8q/) through the script `fmri_align.py`, which itself depends on Alex Huth's tutorial repository: https://github.com/HuthLab/speechmodeltutorial.
To reproduce, clone the tutorial to the root of this repository, and apply the script as described by its usage (e.g. by running `python fmri_align.py -h`).

This reproduction branch assumes the data are all placed into a directory next to this one called `data` (i.e. `../data`).
If you wish to place them elsewhere, the paths in the `*.ini` files of the `cognition_ini` directory must be updated accordingly.

The _Cognition_ experiments (Shain & Schuler, 2021) are defined in the `cognition_ini` directory, with names
corresponding to the various datasets described in the paper. 

In principle, results can be reproduced on UNIX-based systems by navigating to the repository root and invoking the following command:

`make cognition`

However, this will fit hundreds of models sequentially, which will take a long time (probably months).
We therefore recommend either (1) targeted reproduction of the specific models of interest to you or (2) full reproduction on a compute cluster that can fit models in parallel.
Unfortunately, because job schedulers differ substantially between clusters, we do not provide general automation for this use case.
Users will instead need to write their own scripts to schedule all the required jobs.
To this end, note that each `ini` file contains multiple models, each of which is defined by a section prefixed by `model_`.
Everything following this prefix in the section header is used by the system as the model name, with models involving ablated variables additionally using the suffix `!<VAR>` following the model name, where `<VAR>` stands for the name of the ablated variable.

To fit a model, run:

`python -m cdr.bin.fit <INI_FILE> -m <MODEL_NAME>`

For example, if config file `example.ini` contains a model defined in section name `model_CDR_example`, the model can be fitted by running:

`python -m cdr.bin.fit example.ini -m CDR_example`

To evaluate (predict from) a model, run:

`python -m cdr.bin.fit <INI_FILE> -m <MODEL_NAME> -p (train|dev|test)`

Documentation for additional utilities can be viewed by running:

`python -m cdr.bin.help`

For optimal efficiency, each of the above commands should be run as a distinct job.
Results will be placed into a directory next to this one called `results` (i.e. `../results`).

## References
Shain, Cory and Schuler, William (2018). Deconvolutional time series regression: A technique for modeling temporally diffuse effects. _EMNLP18_.

Shain, Cory and Schuler, William (2021). Continuous-time deconvolutional regression for psycholinguistic modeling. _Cognition_.

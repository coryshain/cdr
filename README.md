# Continuous-Time Deconvolutional Regression (CDR)
CDR is a regression technique for modeling temporally diffuse effects (Shain & Schuler, 2018, 2019).

This branch (`npsy`) exists to support reprodiction of results reported in Shain, Blank, et al. (2020).
It locks the repository at a previous state and therefore lacks any subsequent improvements or bug fixes.
Do not use this branch to run regressions on your own data.
Instead, first run the following command from the repository root:

`git checkout -b master`

Installation of dependencies can be managed through Anaconda (https://www.anaconda.com/).
Once you have an Anaconda distribution installed, the software environment can be set up by running the following from the root of this repository:

`conda env create -f npsy.yml`

Once complete, activate the conda environment as follows:

`conda activate npsy`

## _Neuropsychologia_ results

Published results depend on both (1) datasets and (2) models as defined in experiment-specific configuration files.
The data are available on OSF: https://osf.io/eyp8q/.
This reproduction branch assumes the data are all placed into a directory at the repository root called `fmri_data`.
If you wish to place them elsewhere, the paths in the `*.ini` files of the `npsy_ini` directory must be updated accordingly.

The _Neuropsychologia_ experiments (Shain, Blank, et al., 2020) are defined in the `npsy_ini` directory. The files
`lang.ini`, `md.ini`, and `combined.ini` respectively define models for the language network, multiple demand network, and combined language+MD networks.

Results can be reproduced on UNIX-based systems by navigating to the repository root and invoking the following command:

`make npsy`

Windows users without the ``make`` utility will need to either obtain it or run the commands specified in the `npsy` target of the Makefile by hand.

Results will be placed into a directory called `results` at the root of this repository.

## References
Shain, Cory and Schuler, William (2018). Deconvolutional time series regression: A technique for modeling temporally diffuse effects. _EMNLP18_.
Shain, Cory and Schuler, William (2019). Continuous-time deconvolutional regression for psycholinguistic modeling. _PsyArXiv_. [https://doi.org/10.31234/osf.io/whvk5](https://doi.org/10.31234/osf.io/whvk5).
Shain, Cory; Blank, Idan; van Schijndel, Marten; Schuler, William; and Fedorenko, Evelina (2020). fMRI reveals language-specific predictive coding during naturalistic sentence comprehension. _Neuropsychologia_.

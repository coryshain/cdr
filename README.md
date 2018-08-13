# Deconvolutional time series regression (DTSR)
DTSR is a regression technique for modeling temporally diffuse effects (Shain & Schuler, to appear).

This repository contains source code for the `dtsr` Python module as well as support for reproducing published experiments.
Full documentation for the `dtsr` module is available at [http://dtsr.readthedocs.io/en/latest/](http://dtsr.readthedocs.io/en/latest/).

DTSR models can be trained and evaluated using provided utility executables.
Full documentation is provided at the link above.
For example, to fit a DTSR model to the EMNLP18 synthetic data, run the following from the repository root:

`python -m dtsr.bin.train experiments_emnlp18/synth.ini`

Note that some published experiments below also involve fitting LME and GAM models, which require `rpy2` and therefore won't work on Windows systems without some serious hacking.
The `dtsr` module is cross-platform and therefore DTSR models should train regardless of operating system.

## Reproducing published results

Published results depend on both (1) datasets and (2) models as defined in experiment-specific configuration files.
In general, with the exception of a small synthetic dataset, we do not distribute data with this repository.
The datasets used can be provided by email upon request.
Once the data are in hand, the paths to them in the appropriate experiment config files must be updated.

### EMNLP18
The EMNLP18 experiments (Shain & Schuler, to appear) are defined in the `experiments_emnlp18` directory.
The files `natstor_emnlp18.ini`, `dundee_emnlp18.ini`, and `ucl_emnlp18.ini` each contain pointers to data files which will need to be updated by hand.
Once this step is complete, results can be reproduced on UNIX-based systems by navigating to the repository root and invoking the following command:

`make emnlp18`

Windows users without the ``make`` utility will need to either obtain it or run the commands specified in the `emnlp18` target of the Makefile by hand.

## Help and support

For questions, concerns, or data requests, contact Cory Shain ([shain.3@osu.edu](shain.3@osu.edu)).
Bug reports can be logged in the issue tracker on [Github](https://github.com/coryshain/dtsr).


## References
Shain, Cory and Schuler, William (to appear). Deconvolutional time series regression: A technique for modeling temporally diffuse effects. _EMNLP18_.
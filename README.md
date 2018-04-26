# Deconvolutional time series regression (DTSR)
DTSR is a regression technique for modeling temporally diffuse effects.

This repository contains source code for the `dtsr` Python module as well as support for reproducing published experiments.
Full documentation for the `dtsr` module is available at [http://dtsr.readthedocs.io/en/latest/](http://dtsr.readthedocs.io/en/latest/).

Config files that define published experiments are contained in the directories `experiments_reading` and `experiments_synthetic`.
Synthetic data are distributed in this repository.
Reading time data files can be provided upon request.
Once you have a copy of the source data, update the references in the `*.ini` files of the experiment directory to point to the correct locations.

DTSR models can be trained and evaluated using provided utility executables, documented in the API linked to above.
For example, to train a BBVI by-subject slopes model for Natural Stories, run:

`python -m dtsr.bin.train experiments_reading/natstor_bbvi.ini -m DTSR_ss`

To sequentially train all models defined for an experiment (e.g. BBVI on Natural Stories), run

`python -m dtsr.bin.train experiments_reading/natstor_bbvi.ini`

Note that the latter will also try to run baseline LME and GAM models, which require `rpy2` and therefore won't work on Windows systems without some serious hacking.
The `dtsr` module is cross-platform and therefore DTSR models should train regardless of operating system.

For questions, concerns, or data requests, contact Cory Shain ([shain.3@osu.edu](shain.3@osu.edu)).
Bug reports can be logged in the issue tracker on [Github](https://github.com/coryshain/dtsr).

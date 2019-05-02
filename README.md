# Deconvolutional time series regression (DTSR)
DTSR is a regression technique for modeling temporally diffuse effects (Shain & Schuler, 2018).

This branch (emnlp18) exists to support reprodiction of results reported in Shain & Schuler (2018).
It locks the repository at a previous state and therefore lacks any subsequent improvements or bug fixes.
Do not use this branch to run regressions on your own data.
Instead, first run the following command from the repository root:

`git checkout -b master`

Published results depend on both (1) datasets and (2) models as defined in experiment-specific configuration files.
In general, with the exception of small synthetic datasets, we do not distribute data with this repository.
The datasets used can be provided by email upon request.
Once the data are in hand, the paths to them in the appropriate experiment config files must be updated.

### NAACL19
The NAACL19 experiments (Shain, 2019) are defined in the `experiments_naacl19` directory.
The files `natstor_naacl19.ini`, `dundee_naacl19.ini`, and `ucl_naacl19.ini` define the experiments and contain pointers to data.
These experiments depend on the following 12 data files that are not distributed with this repository, but can be provided upon request:

  - `natstor_X.csv`
  - `natstor_y_train.csv`
  - `natstor_y_dev.csv`
  - `natstor_y_test.csv`
  - `dundee_X.csv`
  - `dundee_y_train.csv`
  - `dundee_y_dev.csv`
  - `dundee_y_test.csv`
  - `ucl_X.csv`
  - `ucl_y_train.csv`
  - `ucl_y_dev.csv`
  - `ucl_y_test.csv`
  
The ``*.ini`` files expect to find these resources in the ``experiments_naacl19`` directory of the DTSR repository.
Once the source data are in hand, there are three ways to link them up so you can run the experiments:

  - Create a new directory called ``reading_data`` and place the source data files into it.
  - Create a new directory called ``reading_data`` and put symbolic links in it to source files stored elsewhere.
  - Change the data pointers in the ``*.ini`` files above to point to the locations of the data files.

Once this step is complete, results can be reproduced on UNIX-based systems by navigating to the repository root and invoking the following command:

`make naacl19`

Windows users without the ``make`` utility will need to either obtain it or run the commands specified in the `emnlp18` target of the Makefile by hand.

## Help and support

For questions, concerns, or data requests, contact Cory Shain ([shain.3@osu.edu](shain.3@osu.edu)).
Bug reports can be logged in the issue tracker on [Github](https://github.com/coryshain/dtsr).


## References
Shain, C. A large-scale deconvolutional study of predictability and frequency effects in naturalistic reading. _NAACL 2019_.
Shain, Cory and Schuler, William (2018). Deconvolutional time series regression: A technique for modeling temporally diffuse effects. _EMNLP18_.

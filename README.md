# Continuous-Time Deconvolutional Regressive Neural Networks (CDRNNs)

CDRNNs are a deep neural generalization of continuous-time deconvolutional regression (Shain & Schuler, 2021) that provide flexible and interpretable analyses of complex processes in nature.

This branch of the CDR repository (`cdrnn_journal`) exists to support replication of results reported in Shain & Schuler (2022).
It locks the repository at the state use to generate those results and therefore lacks any subsequent improvements or bug fixes.
Do not use this branch to run regressions on your own data.
Instead, first run the following command from the repository root to switch to the master branch:

    git checkout master

Installation of dependencies can be managed through Anaconda (https://www.anaconda.com).
Once you have an Anaconda distribution installed, the software environment can be set up by running the following from the root of this repository:

`conda env create -f conda_cdrnn.yml`

Once complete, activate the conda environment as follows:

`conda activate cdrnn`

## Main Results

Results depend on both (1) datasets and (2) models as defined in experiment-specific configuration files.
The synthetic, self-paced reading, and fMRI data are available on OSF: https://osf.io/hb5w2/.
The eye-tracking (Dundee) data are not publicly available but can be provided by email upon request (cory.shain@gmail.com).

This reproduction branch assumes the data are all placed into a sibling directory to the repository root called `data`.
If you wish to place them elsewhere, the paths in the `*.ini` files of the `ini` directory must be updated accordingly.

In principle, results can be reproduced on UNIX-based systems by navigating to the repository root and invoking the following command:

`make cdrnn`

However, this will fit models sequentially, which will take a long time (probably months).
We therefore recommend either (1) targeted reproduction of the specific models of interest to you or (2) full reproduction on a compute cluster that can fit models in parallel.
Unfortunately, because job schedulers differ substantially between clusters, we do not provide general automation for this use case.
Users will instead need to write their own scripts to schedule all the required jobs.
To this end, note that each *ini file contains multiple models, each of which is defined by a section prefixed by `model_`.
Everything following this prefix in the section header is used by the system as the model name, with models involving ablated variables additionally using the suffix `!<VAR>` following the model name, where `<VAR>` stands for the name of the ablated variable.

To fit a model, run:

`python -m cdr.bin.train <INI_FILE> -m <MODEL_NAME>`

For example, if config file `example.ini` contains a model defined in section name `model_CDRNN_example`, the model can be fitted by running:

`python -m cdr.bin.train example.ini -m CDRNN_example`

To evaluate (predict from) a model, run:

`python -m cdr.bin.predict <INI_FILE> -m <MODEL_NAME> -p (train|dev|test)`

Documentation for additional utilities can be viewed by running:

`python -m cdr.bin.help`

Full documentation and API are available here: https://cdr.readthedocs.io/en/latest/.

For optimal efficiency, each of the above commands should be run as a distinct job.
Results will be placed into a directory called `results` at the root of this repository.

## References
Shain, Cory and Schuler, William (2018). Deconvolutional time series regression: A technique for modeling temporally diffuse effects. _EMNLP18_.

Shain, Cory and Schuler, William (2021). Continuous-time deconvolutional regression for psycholinguistic modeling. _Cognition_.

Shain, Cory (2021). CDRNN: Discovering complex dynamics in human language processing. _ACL21_.

Shain, Cory and Schuler, William (2022). A Deep Learning Approach to Analyzing Continuous-Time Systems. _PsyArXiv_.


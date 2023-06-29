# Continuous-Time Deconvolutional Regressive Neural Networks (CDRNNs)

CDRNNs are a deep neural generalization of continuous-time deconvolutional regression (Shain & Schuler, 2021) that provide flexible and interpretable analyses of complex processes in nature.

This branch of the CDR repository (`freq_pred`) exists to support replication of results reported in Shain (2023).
It locks the repository at the state use to generate those results and therefore lacks any subsequent improvements or bug fixes.
Do not use this branch to run regressions on your own data.
Instead, first run the following command from the repository root to switch to the master branch:

    git checkout master

Installation of dependencies can be managed through Anaconda (https://www.anaconda.com).
Once you have an Anaconda distribution installed, the software environment can be set up by running the following from the root of this repository:

`conda env create -f conda_cdr.yml`

Once complete, activate the conda environment as follows:

`conda activate cdr`

## Main Results

Results depend on both (1) datasets and (2) models as defined in experiment-specific configuration files.
Data are available for public download from OSF: https://osf.io/8v5qb/.

This reproduction branch assumes the data are all placed into a sibling directory to the repository root called `data`.
If you wish to place them elsewhere, the paths in the `*.ini` files of the `ini` directory must be updated accordingly.

Individual models can be fitted by running the following command from the repository root:

    python -m cdr.bin.train ini_owt/DATASET.ini -m MODEL_NAME

However, this study involved hundreds of fitted models that could take weeks to fit sequentially.
Therefore, this branch assumes reproduction on a SLURM-based compute cluster.

To generate training batch scripts to submit to the job scheduler, run:

    python -m cdr.bin.make_jobs ini_owt/*.ini CONFIG_OPTIONS

where `CONFIG_OPTIONS` stands in for any scheduler options you want to include. For details, run:

    python -m cdr.bin.make_jobs -h

To generate test set prediction batch scripts to submit to the job scheduler, run:

    python -m cdr.bin.make_jobs ini_owt/*.ini CONFIG_OPTIONS -j predict -p test

To generate significance test batch scripts to submit to the job scheduler, run:

    python make_test_jobs.py

To generate plotthing batch scripts to submit to the job scheduler, run:

    python make_plot_jobs.py

Note that you may need to edit the data and/or output paths in the preceding two scripts.

To submit all existing batch scripts (suffix `*.pbs`), run:

    ./qsub.sh *.pbs

Models, plots and testing results will appear in a sibling directory called `results/cdrnn_freq_pred_owt`.
To check model likelihoods and tabulate significance tests, respectively run:

    python -m cdr.bin.error_table ini_owt/DATASET.ini -m ll
    python -m cdr.bin.signif_table PATH/TO/TEST/RESULTS -s signif_table.yml

Full documentation and API are available here: https://cdr.readthedocs.io/en/latest/.

## References
Shain, Cory and Schuler, William (2018). Deconvolutional time series regression: A technique for modeling temporally diffuse effects. _EMNLP18_.

Shain, Cory and Schuler, William (2021). Continuous-time deconvolutional regression for psycholinguistic modeling. _Cognition_.

Shain, Cory (2021). CDRNN: Discovering complex dynamics in human language processing. _ACL21_.

Shain, Cory and Schuler, William (2022). A Deep Learning Approach to Analyzing Continuous-Time Systems. _PsyArXiv_.

Shain, Cory (2023). Word frequency and predictability effects dissociate in naturalistic reading. _PsyArXiv_.


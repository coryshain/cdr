# Continuous-Time Deconvolutional Regressive Neural Networks (CDRNNs)

CDRNNs are a deep neural generalization of continuous-time deconvolutional regression (Shain & Schuler, 2021) that provide flexible and interpretable analyses of complex processes in nature.

This branch of the CDR repository (`pred_fn`) exists to support replication of results reported in Shain et al. (2023).
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
Data are available for public download from OSF: https://osf.io/6wvqe/.
The data repository should be downloaded and unzipped to a folder called ``data`` at the root of this directory.
The configuration files defining all models are located in the `pred_fn_ini` folder.

Individual models can be fitted by running the following command from the repository root:

    python -m cdr.bin.train pred_fn_ini/DATASET.ini -m MODEL_NAME

However, this study involved hundreds of fitted models that could take weeks to fit sequentially.
Therefore, this branch assumes reproduction on a SLURM-based compute cluster.

To generate training batch scripts to submit to the job scheduler, run:

    python -m cdr.bin.make_jobs pred_fn_ini/*.ini CONFIG_OPTIONS

where `CONFIG_OPTIONS` stands in for any scheduler options you want to include. For details, run:

    python -m cdr.bin.make_jobs -h

To generate test set prediction batch scripts to submit to the job scheduler, run:

    python -m cdr.bin.make_jobs pred_fn_ini/*.ini CONFIG_OPTIONS -j predict -p test

To generate significance test batch scripts to submit to the job scheduler, run:

    python make_test_jobs.py

To generate plotthing batch scripts to submit to the job scheduler, run:

    python make_plot_jobs.py

Note that you may need to edit the source code in the preceding two scripts to allow them to work in your local runtime environment.
See comments in the code for details.

To submit all existing batch scripts (suffix `*.pbs`), run:

    ./qsub.sh *.pbs

Models, plots and testing results will appear in a folder called `results` in this directory.
To check model likelihoods and tabulate significance tests, respectively run:

    python -m cdr.bin.error_table pred_fn_ini/DATASET.ini -m ll
    python -m cdr.bin.signif_table results/signif/DATASET -s signif_table.yml

Full documentation and API are available here: https://cdr.readthedocs.io/en/latest/.

## Supplementary GAM analyses

To preprocess data for GAM fitting to a given dataset, run:

    python -m cdr.gam.get_gam_data pred_fn_ini/<DATASET>.ini

To generate regression scripts, run:

    python -m cdr.gam.make_gam_jobs

This will generate R scripts (`*.R`) and SLURM batch scripts (`*.pbs`) corresponding to each GAM model in the analysis, saved in a directory called `gam_scripts`.

To generate plots from fitted GAM models, run:

    python -m cdr.gam.plot

To generate SLURM batch jobs for hypothesis tests, run:

    python -m cdr.gam.make_test_jobs

This will generate several batch scripts (`*.pbs`) in the current directory that can be submitted to job scheduler.

To generate performance tables for a given dataset, run:

    python -m cdr.gam.gam_error_table <NAME_OF_CORPUS>

## Supplementary analyses of data from Brothers & Kuperberg (2021)

To collate Brothers & Kuperberg's (BK21's) stimuli for analysis, run:

    python -m cdr.bk21.get_items

To compute unigram, GPT2, and GloVe features for these items run (respectively):

    python -m cdr.bk21.add_{unigram,gpt,glove}

To merge these textual features with the SPR and naming data for analysis, run (respectively):

    python -m cdr.bk21.merge_{spr,naming}

To generate regression scripts, run:

    python -m cdr.bk21.make_bk21_jobs

This will generate R scripts (`*.R`) and SLURM batch files (`*.pbs`) corresponding to each model in the analysis, saved in a directory called `bk21_scripts`.

To generate plots (requires that all regressions have completed), run:

    python -m cdr.bk21.plot

To generate SLURM batch jobs for hypothesis tests, run:

    python -m cdr.bk21.make_test_jobs

This will generate several batch scripts (`*pbs`) in the current directory that can be submitted to the job scheduler.

## References
Shain, Cory and Schuler, William (2018). Deconvolutional time series regression: A technique for modeling temporally diffuse effects. _EMNLP18_.

Brothers, Trevor and Kuperberg, Gina (2021). Word predictability effects are linear, not logarithmic: Implications for probabilistic models of sentence comprehension. _JML_.

Shain, Cory and Schuler, William (2021). Continuous-time deconvolutional regression for psycholinguistic modeling. _Cognition_.

Shain, Cory (2021). CDRNN: Discovering complex dynamics in human language processing. _ACL21_.

Shain, Cory and Schuler, William (2022). A Deep Learning Approach to Analyzing Continuous-Time Systems. _arXiv_.

Shain, Cory and Meister, Clara and Pimentel, Tiago and Cotterell, Ryan and Levy, Roger (2023). Large-Scale Evidence for Logarithmic Effects of Word Predictability on Reading Time. _PsyArXiv_. https://psyarxiv.com/4hyna/.


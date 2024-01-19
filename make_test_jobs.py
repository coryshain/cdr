import sys
import os
import argparse
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

argparser = argparse.ArgumentParser('Generate SLURM jobs for model comparison')
argparser.add_argument('-c', '--cutoff95', action='store_true', help='Run on cutoff95 models. Otherwise run in main models.')
args = argparser.parse_args()

cutoff95 = ' -c' if args.cutoff95 else ''

base = """#!/bin/bash
#
#SBATCH --job-name=%s
#SBATCH --output="%s-%%N-%%j.out"
#SBATCH --time=48:00:00
#SBATCH --mem=16gb
#SBATCH --ntasks=4
"""

assert os.path.exists('config.yml'), 'Repository has not yet been initialized. First run `python -m initialize`.'
with open('config.yml', 'r') as f:
    repo_cfg = yaml.load(f, Loader=Loader)

if repo_cfg['singularity_path']:
    wrapper = '\nsingularity exec --nv %s bash -c "%%s"' % repo_cfg['singularity_path']
else:
    wrapper = '\n%s'

datasets = [
    'brown.fdur',
    'dundee.fdurSPsummed',
    'dundee.fdurFP',
    'dundee.fdurGP',
    'geco.fdurFP',
    'geco.fdurGP',
    'natstor.fdur',
    'natstormaze.rt',
    'provo.fdurSPsummed',
    'provo.fdurFP',
    'provo.fdurGP',
    'all'
]

comparisons = [
    'main-gpt_v_main',
    'main-unigramsurpOWT_v_main',
    'main-unigramsurpOWT-gpt_v_main-unigramsurpOWT',
    'main-unigramsurpOWT-gpt_v_main-gpt',
    'nointeraction_v_yesinteraction',
    'dist-unigramsurpOWTmu_v_dist',
    'dist-unigramsurpOWTsigma_v_dist',
    'dist-unigramsurpOWTbeta_v_dist',
    'dist-gptmu_v_dist',
    'dist-gptsigma_v_dist',
    'dist-gptbeta_v_dist',
    'main_v_bigram',
    'bigram-gpt_v_bigram',
    'bigram-unigramsurpOWT_v_bigram',
    'bigram_v_trigram',
]

for dataset in datasets:
    for comparison in comparisons:
        job_name = '%s_%s_test' % (dataset, comparison)
        job_str = 'python3 test_freq_pred.py %s %s%s' % (dataset, comparison, cutoff95)
        job_str = wrapper % job_str
        job_str = base % (job_name, job_name) + job_str
        with open(job_name + '.pbs', 'w') as f:
            f.write(job_str)

if not cutoff95:
    # Shain 2019 comparison
    
    comparisons = [
        'log-unigramsurpOWT_v_log',
        'log-gpt_v_log',
        'log-unigramsurpOWT-gpt_v_log-unigramsurpOWT',
        'log-unigramsurpOWT-gpt_v_log-gpt',
        'homoscedastic-unigramsurpOWT_v_homoscedastic',
        'homoscedastic-gpt_v_homoscedastic',
        'homoscedastic-unigramsurpOWT-gpt_v_homoscedastic-unigramsurpOWT',
        'homoscedastic-unigramsurpOWT-gpt_v_homoscedastic-gpt',
        'additive-unigramsurpOWT_v_additive',
        'additive-gpt_v_additive',
        'additive-unigramsurpOWT-gpt_v_additive-unigramsurpOWT',
        'additive-unigramsurpOWT-gpt_v_additive-gpt',
        'loghomoscedastic-unigramsurpOWT_v_loghomoscedastic',
        'loghomoscedastic-gpt_v_loghomoscedastic',
        'loghomoscedastic-unigramsurpOWT-gpt_v_loghomoscedastic-unigramsurpOWT',
        'loghomoscedastic-unigramsurpOWT-gpt_v_loghomoscedastic-gpt',
        'logadditive-unigramsurpOWT_v_logadditive',
        'logadditive-gpt_v_logadditive',
        'logadditive-unigramsurpOWT-gpt_v_logadditive-unigramsurpOWT',
        'logadditive-unigramsurpOWT-gpt_v_logadditive-gpt',
        'homoscedasticadditive-unigramsurpOWT_v_homoscedasticadditive',
        'homoscedasticadditive-gpt_v_homoscedasticadditive',
        'homoscedasticadditive-unigramsurpOWT-gpt_v_homoscedasticadditive-unigramsurpOWT',
        'homoscedasticadditive-unigramsurpOWT-gpt_v_homoscedasticadditive-gpt',
        'loghomoscedasticadditive-unigramsurpOWT_v_loghomoscedasticadditive',
        'loghomoscedasticadditive-gpt_v_loghomoscedasticadditive',
        'loghomoscedasticadditive-unigramsurpOWT-gpt_v_loghomoscedasticadditive-unigramsurpOWT',
        'loghomoscedasticadditive-unigramsurpOWT-gpt_v_loghomoscedasticadditive-gpt',
    ]
    for dataset in ('dundee.fdurGP', 'natstor.fdur'):
        for comparison in comparisons:
            job_name = '%s_%s_test' % (dataset, comparison)
            job_str = 'python3 test_freq_pred.py %s %s' % (dataset, comparison)
            job_str = wrapper % job_str
            job_str = base % (job_name, job_name) + job_str
            with open(job_name + '.pbs', 'w') as f:
                f.write(job_str)


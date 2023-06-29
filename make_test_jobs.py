import sys

base = """#!/bin/bash
#
#SBATCH --job-name=%s
#SBATCH --output="%s-%%N-%%j.out"
#SBATCH --time=48:00:00
#SBATCH --mem=16gb
#SBATCH --ntasks=4
"""

# Change this to reflect how your system runs bash executables
wrapper = '\nsingularity exec --nv ../singularity_images/tf-latest-gpu.simg bash -c "%s"'

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
        job_str = 'python3 test_freq_pred.py %s %s' % (dataset, comparison)
        job_str = wrapper % job_str
        job_str = base % (job_name, job_name) + job_str
        with open(job_name + '.pbs', 'w') as f:
            f.write(job_str)


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


import sys

base = """#!/bin/bash
#
#SBATCH --job-name=%s
#SBATCH --output="%s-%%N-%%j.out"
#SBATCH --time=48:00:00
#SBATCH --mem=16gb
#SBATCH --ntasks=1
"""

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
    'null_v_freqonly',
    'null_v_predonly',
    'freqonly_v_both',
    'predonly_v_both',
    'both_v_interaction',
]

for dataset in datasets:
    for comparison in comparisons:
        job_name = 'lme_%s_%s_test' % (dataset, comparison)
        job_str = 'python -m cdr.lme.test %s %s' % (dataset, comparison)
        job_str = wrapper % job_str
        job_str = base % (job_name, job_name) + job_str
        with open(job_name + '.pbs', 'w') as f:
            f.write(job_str)


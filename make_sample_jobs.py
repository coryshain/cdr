import sys
import os
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

base = """#!/bin/bash
#
#SBATCH --job-name=%s
#SBATCH --output="%s-%%N-%%j.out"
#SBATCH --time=8:00:00
#SBATCH --mem=8gb
#SBATCH --ntasks=4
"""

assert os.path.exists('config.yml'), 'Repository has not yet been initialized. First run `python -m initialize`.'
with open('config.yml', 'r') as f:
    repo_cfg = yaml.load(f, Loader=Loader)

ini_dir = repo_cfg['ini_path']
if repo_cfg['singularity_path']:
    wrapper = '\nsingularity exec --nv %s bash -c "%%s"' % repo_cfg['singularity_path']
else:
    wrapper = '\n%s'

datasets = [
    'brown',
    'dundee',
    'geco',
    'natstor',
    'natstormaze',
    'provo',
]

N_SAMPLES = 100

for dataset in datasets:
    job_name = '%s_sample' % dataset
    job_str = 'python3 -m cdr.bin.sample %s/%s.ini -m main -n %d -e -p test' % (ini_dir, dataset, N_SAMPLES)
    job_str = wrapper % job_str
    job_str = base % (job_name, job_name) + job_str
    with open(job_name + '.pbs', 'w') as f:
        f.write(job_str)


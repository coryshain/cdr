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

models = [
    'ngram',
    'pcfg',
    'gpt',
    'gptj',
    'gpt3',
    'cloze'
]

functions = [
    'prob',
    '0.50',
    '0.75',
    '1.00',
    '1.33',
    '2.00',
    '',
]

comparisons_nosurp = []
for model in models + ['allLM']:
    for fn in functions:
        s = 'nosurp_v_%s%s' % (model, fn)
        comparisons_nosurp.append(s)

comparisons_fn = []
for i, fn1 in enumerate(functions):
    for model in models + ['allLM']:
        if i < len(functions) - 1:
            for fn2 in functions[i+1:]:
                s = '%s%s_v_%s%s' % (model, fn1, model, fn2)
                comparisons_fn.append(s)

comparisons = comparisons_nosurp + comparisons_fn

for dataset in datasets:
    for comparison in comparisons:
        if 'cloze' not in comparison or dataset.startswith('provo'):
            job_name = 'gam_%s_%s_test' % (dataset, comparison)
            job_str = 'python -m cdr.gam.test %s %s' % (dataset, comparison)
            job_str = wrapper % job_str
            job_str = base % (job_name, job_name) + job_str
            with open(job_name + '.pbs', 'w') as f:
                f.write(job_str)


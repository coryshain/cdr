import sys
import argparse

base = """#!/bin/bash
#
#SBATCH --job-name=%s
#SBATCH --output="%s-%%N-%%j.out"
#SBATCH --time=96:00:00
#SBATCH --mem=16gb
#SBATCH --ntasks=4
"""

# Wrapper is useful for running in singularity environments (you'll have to change the path to the image).
# If not using singularity, set wrapper to '%s' (no-op)
wrapper = '\nsingularity exec --nv ../singularity_images/tf-latest-gpu.simg bash -c "%s"' # REPLACE THESE PATHS

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

datasets_normal = []
for dataset in datasets:
    if dataset == 'all':
        dataset = 'all_normal'
    else:
        dataset = '%s_normal.%s' % tuple(dataset.split('.'))
    datasets_normal.append(dataset)

models = [
    'cloze',
    'ngram',
    'pcfg',
    'gpt',
    'gptj',
    'gpt3',
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

composite = [
    ('lin', 'sublin'),
    ('lin', 'suplin'),
    ('notsublin', 'sublin'),
    ('notsuplin', 'suplin'),
]

comparisons_nosurp = []
comparisons_nosurp_normal = []
for model in models + ['allLM']:
    for fn in functions:
        s = 'nosurp_v_%s%s' % (model, fn)
        comparisons_nosurp.append(s)
        if model == 'gpt':
            comparisons_nosurp_normal.append(s)

comparisons_fn = []
comparisons_fn_normal = []
for i, fn1 in enumerate(functions):
    for model in models + ['allLM']:
        if i < len(functions) - 1:
            for fn2 in functions[i+1:]:
                s = '%s%s_v_%s%s' % (model, fn1, model, fn2)
                comparisons_fn.append(s)
                if model == 'gpt':
                    comparisons_fn_normal.append(s)

comparisons_composite = []
comparisons_composite_normal = []
for m in models + ['allLM']:
    for composite_comparison in composite:
        a, b = composite_comparison
        s = '%s%s_v_%s%s' % (m, a, m, b)
        comparisons_composite.append(s)
        if m == 'gpt':
            comparisons_composite_normal.append(s)

comparisons_model = []
for i, m1 in enumerate(models):
    if i < len(models) - 1:
        for m2 in models[i+1:]:
            s = '%s_v_%s' % (m1, m2)
            comparisons_model.append(s)
comparisons_model.append('gpt_v_gptpcfg')

comparisons_surpproblin = []
for m in ['gpt']:
    for h0 in ('prob', '1.00'):
       s = '%s%s_v_%ssurpproblinear' % (m, h0, m)
       comparisons_surpproblin.append(s)

comparisons_dist = []
for m in ['gpt']:
    s = '%snormal_v_%s' % (m, m)
    comparisons_dist.append(s)

comparisons = comparisons_nosurp + comparisons_fn + comparisons_composite + comparisons_model + comparisons_surpproblin + comparisons_dist

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Generate SLURM batch scripts for permutation tests')
    argparser.add_argument('-r', '--results_path', default='results', help='Path to directory containing modeling results')
    args = argparser.parse_args()
    results_path = args.results_path
    
    # Main analyses
    for dataset in datasets:
        for comparison in comparisons:
            if dataset.startswith('provo') or 'cloze' not in comparison:
                job_name = '%s_%s_test' % (dataset, comparison)
                job_str = 'python3 test_surp_lin.py %s %s -r %s' % (dataset, comparison, results_path)
                job_str = wrapper % job_str
                job_str = base % (job_name, job_name) + job_str
                with open(job_name + '.pbs', 'w') as f:
                    f.write(job_str)
    
    # Normal error
    model = 'gpt'
    for dataset in datasets_normal:
        for comparison in comparisons_nosurp_normal + comparisons_fn_normal + comparisons_composite_normal:
            job_name = '%s_%s_test' % (dataset, comparison)
            job_str = 'python3 test_surp_lin.py %s %s -r %s' % (dataset, comparison, results_path)
            job_str = wrapper % job_str
            job_str = base % (job_name, job_name) + job_str
            with open(job_name + '.pbs', 'w') as f:
                f.write(job_str)
            
    # Word skipping
    model = 'gpt'
    comparisons_skip = []
    functions = [
        'prob',
        '1.00',
        '',
    ]
    for model in ['gpt']:
        for fn in functions:
            s = 'nosurp_v_%s%s' % (model, fn)
            comparisons_skip.append(s)
    for i, fn1 in enumerate(functions):
        for model in ['gpt']:
            if i < len(functions) - 1:
                for fn2 in functions[i+1:]:
                    s = '%s%s_v_%s%s' % (model, fn1, model, fn2)
                    comparisons_skip.append(s)
    for dataset in datasets_skip:
        for comparison in comparisons_skip:
            job_name = '%s_%s_test' % (dataset, comparison)
            job_str = 'python3 test_surp_lin_curr.py %s %s -r %s' % (dataset, comparison, results_path)
            job_str = wrapper % job_str
            job_str = base % (job_name, job_name) + job_str
            with open(job_name + '.pbs', 'w') as f:
                f.write(job_str)

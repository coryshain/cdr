import sys

base = """#!/bin/bash
#
#SBATCH --job-name=%s
#SBATCH --output="%s-%%N-%%j.out"
#SBATCH --time=96:00:00
#SBATCH --mem=16gb
#SBATCH --ntasks=4
"""

wrapper = '\nsingularity exec --nv ../singularity_images/tf-latest-gpu.simg bash -c "cd /net/vast-storage.ib.cluster/scratch/vast/evlab/cshain/cdr_surp_lin; %s"' # REPLACE THESE PATHS

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

for dataset in datasets:
    for comparison in comparisons:
        if dataset.startswith('provo') or 'cloze' not in comparison:
            job_name = '%s_%s_test' % (dataset, comparison)
            job_str = 'python3 test_surp_lin.py %s %s' % (dataset, comparison)
            job_str = wrapper % job_str
            job_str = base % (job_name, job_name) + job_str
            with open(job_name + '.pbs', 'w') as f:
                f.write(job_str)

model = 'gpt'
for dataset in datasets_normal:
    for comparison in comparisons_nosurp_normal + comparisons_fn_normal + comparisons_composite_normal:
        job_name = '%s_%s_test' % (dataset, comparison)
        job_str = 'python3 test_surp_lin.py %s %s' % (dataset, comparison)
        job_str = wrapper % job_str
        job_str = base % (job_name, job_name) + job_str
        with open(job_name + '.pbs', 'w') as f:
            f.write(job_str)
        

# if len(sys.argv) > 1:
#     partition = sys.argv[1]
# else:
#     partition = 'dev'
# 
# models = ('prob_h0', '0.50_h0', '0.75_h0', '1.00_h0', '1.33_h0', '2.00_h0', '')
# for cfg in ('all', 'brown.ini', 'dundee.ini', 'geco.ini', 'natstor.ini', 'natstormaze.ini', 'provo.ini'):
#     dataset = cfg.replace('.ini', '')
#     cliargs = ' -o ../results/surp_lin/signif/%s' % dataset
#     if cfg == 'all':
#         cfg = '{brown,dundee,geco,natstor,natstormaze,provo}.ini'
#         cliargs += ' -P'
#     for comp in [
#         ('cloze', 'pcfg'),
#         ('cloze', 'ngram'),
#         ('cloze', 'gpt'),
#         ('cloze', 'gptj'),
#         ('cloze', 'gpt3'),
#         ('pcfg', 'ngram'),
#         ('pcfg', 'gpt'),
#         ('pcfg', 'gptj'),
#         ('pcfg', 'gpt3'),
#         ('ngram', 'gpt'),
#         ('ngram', 'gptj'),
#         ('ngram', 'gpt3'),
#         ('gpt', 'gptj'),
#         ('gpt', 'gpt3'),
#         ('gptj', 'gpt3'),
#         ('gpt', 'gptpcfg'),
#         ('gptprob_h0', 'gptsurpproblin'),
#         ('gpt1.00_h0', 'gptsurpproblin'),
#     ]:
#         if cfg.startswith('provo') or 'cloze' not in comp:
#             a, b = comp
#             job_name = '%s_%s_v_%s_test' % (dataset, a, b)
#             job_str = 'python3 -m cdr.bin.test ini/%s -m CDR_%s CDR_%s -M loglik -p %s%s' % (cfg, a, b, partition, cliargs)
#             job_str = wrapper % job_str
#             job_str = base % (job_name, job_name) + job_str
#             with open(job_name + '.pbs', 'w') as f:
#                 f.write(job_str)
#         
#     for surp in ('ngram', 'pcfg', 'gpt', 'gptj', 'gpt3', 'cloze'):
#         if cfg.startswith('provo') or surp != 'cloze':
#             for i in range(len(models)):
#                 suff1 = models[i]
#                 job_name = '%s_nosurp_v_%s%s_test' % (dataset, surp, suff1)
#                 job_str = 'python3 -m cdr.bin.test ini/%s -m CDR_nosurp CDR_%s%s -M loglik -p %s%s' % (cfg, surp, suff1, partition, cliargs)
#                 job_str = wrapper % job_str
#                 job_str = base % (job_name, job_name) + job_str
#                 with open(job_name + '.pbs', 'w') as f:
#                     f.write(job_str)
#                 for j in range(i+1, len(models)):
#                     suff2 = models[j]
#                     job_name = '%s_%s%s_v_%s%s_test' % (dataset, surp, suff1, surp, suff2)
#                     job_str = 'python3 -m cdr.bin.test ini/%s -m CDR_%s%s CDR_%s%s -M loglik -p %s%s' % (cfg, surp, suff1, surp, suff2, partition, cliargs)
#                     job_str = wrapper % job_str
#                     job_str = base % (job_name, job_name) + job_str
#                     with open(job_name + '.pbs', 'w') as f:
#                         f.write(job_str)
#     
# # Normal error models
# for cfg in ('all_normal', 'brown_normal.ini', 'dundee_normal.ini', 'geco_normal.ini', 'natstor_normal.ini', 'natstormaze_normal.ini', 'provo_normal.ini'):
#     dataset = cfg.replace('.ini', '')
#     cliargs = ' -o ../results/surp_lin/signif/%s' % dataset
#     if cfg == 'all':
#         cfg = '{brown,dundee,geco,natstor,natstormaze,provo}_normal.ini'
#         cliargs += ' -P'
#     for surp in ('gpt',):
#         for i in range(len(models)):
#             suff1 = models[i]
#             job_name = '%s_nosurp_v_%s%s_test' % (dataset, surp, suff1)
#             job_str = 'python3 -m cdr.bin.test ini/%s -m CDR_nosurp CDR_%s%s -M loglik -p %s%s' % (cfg, surp, suff1, partition, cliargs)
#             job_str = wrapper % job_str
#             job_str = base % (job_name, job_name) + job_str
#             with open(job_name + '.pbs', 'w') as f:
#                 f.write(job_str)
#             for j in range(i+1, len(models)):
#                 suff2 = models[j]
#                 job_name = '%s_%s%s_v_%s%s_test' % (dataset, surp, suff1, surp, suff2)
#                 job_str = 'python3 -m cdr.bin.test ini/%s -m CDR_%s%s CDR_%s%s -M loglik -p %s' % (cfg, surp, suff1, surp, suff2, partition)
#                 job_str = wrapper % job_str
#                 job_str = base % (job_name, job_name) + job_str
#                 with open(job_name + '.pbs', 'w') as f:
#                     f.write(job_str)
#  

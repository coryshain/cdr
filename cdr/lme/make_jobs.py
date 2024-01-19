import os
import stat
import argparse

argparser = argparse.ArgumentParser('Generate R scripts for LME regressions')
args = argparser.parse_args()

for eval_type in ('insample', 'outofsample'):
    R_base = '''#!/usr/bin/env Rscript

library(lme4)

if (!dir.exists('{{results_path}}/{{corpus}}')) {{{{
  dir.create('{{results_path}}/{{corpus}}', recursive=TRUE)
}}}}

# Paths
{paths}

# Data loading
{data}
df_train = df_train[is.finite(df_train${{dv}}),]
df_train[is.na(df_train)] = 0

df_test = read.table("lme_data/{{corpus}}_test.csv", sep=" ", header=TRUE)
df_test = df_test[is.finite(df_test${{dv}}),]
for (col in c('subject', 'docid_sentid_sentpos')) {{{{
    df_train[[col]] = as.factor(df_train[[col]])
    df_test[[col]] = as.factor(df_test[[col]])
}}}}
df_test[is.na(df_test)] = 0

# Fitting
form = {{form}}
m = lmer(form, data=df_train, REML=FALSE, control=lmerControl(optimizer='bobyqa', optCtrl=list(maxfun=1e6)))
save(m, file=model_path)

# Summary
sink(summary_path)
print(summary(m))
sink()

# Evaluation
out = predict(m, newdata=df_test, allow.new.levels=TRUE)
obs = df_test[['{{dv}}']]
err = obs - out
sqerr = err^2
mse = mean(sqerr)
variance = mean(residuals(m, type='response')^2)
ll = dnorm(err, mean=0, sd=sqrt(variance), log=TRUE)
ll_summed = sum(ll)
eval_str = paste0(
    '==================================================\\n',
    'LME regression\\n\\n',
    'Model path: ', model_path, '\\n',

    'MODEL EVALUATION STATISTICS:\\n',
    '  Loglik: ', ll_summed, '\\n',
    '  MSE: ', mse, '\\n',
    '==================================================\\n'
)

cat(eval_str, file=stderr())

write.table(eval_str, eval_path, row.names=FALSE, col.names=FALSE)
output = data.frame(LMEobs=obs, LMEpred=out, LMEerr=err, LMEloglik=ll)
write.table(output, output_path, row.names=FALSE, col.names=TRUE, sep=',')
'''

    paths_is = """
model_path = '{results_path}/{corpus}/{corpus}_{dv}_{crit_var_set_name}_insample.Rdata'
summary_path = '{results_path}/{corpus}/{corpus}_{dv}_{crit_var_set_name}_insample_summary.txt'
eval_path = '{results_path}/{corpus}/{corpus}_{dv}_{crit_var_set_name}_insample_eval.txt'
output_path = '{results_path}/{corpus}/{corpus}_{dv}_{crit_var_set_name}_insample_output.csv'"""

    paths_oos = """
model_path = '{results_path}/{corpus}/{corpus}_{dv}_{crit_var_set_name}_outofsample.Rdata'
summary_path = '{results_path}/{corpus}/{corpus}_{dv}_{crit_var_set_name}_outofsample_summary.txt'
eval_path = '{results_path}/{corpus}/{corpus}_{dv}_{crit_var_set_name}_outofsample_eval.txt'
output_path = '{results_path}/{corpus}/{corpus}_{dv}_{crit_var_set_name}_outofsample_output.csv'"""

    data_is = 'df_train = read.table("lme_data/{corpus}_train-dev-test.csv", sep=" ", header=TRUE)'
    data_oos = 'df_train = read.table("lme_data/{corpus}_train-dev.csv", sep=" ", header=TRUE)'

    R_test = '''#!/usr/bin/env Rscript

library(lme4)

# Paths
m1_path = '{results_path}/{corpus}/{corpus}_{dv}_{m1}_insample.Rdata'
m2_path = '{results_path}/{corpus}/{corpus}_{dv}_{m2}_insample.Rdata'
summary_path = '{results_path}/{corpus}/{corpus}_{dv}_test_{m1}_v_{m2}_insample.txt'

m1 = get(load(m1_path))
m2 = get(load(m2_path))

# Summary
sink(summary_path)
print(summary(m2))
print(anova(m1, m2))
sink()
'''
    
    bash_fit = '''#!/bin/bash
#
#SBATCH --job-name={job_name}
#SBATCH --output="{job_name}-%N-%j.out"
#SBATCH --time=48:00:00
#SBATCH --mem={mem}gb
#SBATCH --ntasks=4

lme_scripts/{job_name}.R
'''
    
    bash_test_base = '''#!/bin/bash
#
#SBATCH --job-name={{job_name}}
#SBATCH --output="{{job_name}}-%N-%j.out"
#SBATCH --time=48:00:00
#SBATCH --mem={{mem}}gb
#SBATCH --ntasks=4

{cmd}
'''
    
    bash_test_cmd_is = 'lme_scripts/{job_name}.R{dataset}{dv}{comparison}'  # dataset, dv, and comparison should be empty, included for consistent API
    bash_test_cmd_oos = 'python -m cdr.lme.test {dataset}.{dv} {comparison}'
    
    if eval_type == 'outofsample':
       R_format_kwargs = dict(
           paths=paths_oos,
           data=data_oos
       )
       bash_format_kwargs = dict(
           cmd=bash_test_cmd_oos,
       )
    else:
       R_format_kwargs = dict(
           paths=paths_is,
           data=data_is
       )
       bash_format_kwargs = dict(
           cmd=bash_test_cmd_is
       )
    R = R_base.format(**R_format_kwargs)
    bash_test = bash_test_base.format(**bash_format_kwargs)
    
    common = ['endofsentence', 'wlen']
    
    baseline_preds = {
        'brown': common,
        'dundee': common + ['inregression', 'endofline', 'endofscreen', 'wdelta'],
        'geco': common + ['wdelta'],
        'natstor': common,
        'natstormaze': common + ['incorrect'],
        'provo': common + ['inregression', 'wdelta']
    }
    dvs = {
        'brown': ['fdur'],
        'dundee': ['fdurSPsummed', 'fdurFP', 'fdurGP'],
        'geco': ['fdurFP', 'fdurGP'],
        'natstor': ['fdur'],
        'natstormaze': ['rt'],
        'provo': ['fdurSPsummed', 'fdurFP', 'fdurGP'],
    }
    
    crit_var_sets = {
        'null': [],
        'freqonly': ['unigramsurpOWT'],
        'predonly': ['gpt'],
        'both': ['unigramsurpOWT', 'gpt'],
        'interaction': ['unigramsurpOWT', 'gpt', 'unigramsurpOWT:gpt'],
    }
    crit_vars_all = []
    for x in crit_var_sets:
        for y in crit_var_sets[x]:
            if y not in crit_vars_all:
                crit_vars_all.append(y)
    comparisons = [
        'null_v_freqonly',
        'null_v_predonly',
        'freqonly_v_both',
        'predonly_v_both',
        'both_v_interaction',
    ]
    
    N_SPILL = 1
    
    if os.path.exists('lme_results_path.txt'):
        with open('lme_results_path.txt') as f:
            for line in f:
               line = line.strip()
               if line:
                   results_path = line
                   break
    else:
        results_path = 'results/lme'
    
    for corpus in dvs:
        dataset_basepath = 'lme_data/%s' % corpus
        for dv in dvs[corpus]:
            # FITTING
            for crit_var_set_name in crit_var_sets:
                crit_var_set = crit_var_sets[crit_var_set_name]
                controls = baseline_preds[corpus].copy()
                if dv != 'fdurSPsummed':
                    controls = [x for x in controls if x != 'inregression']
                preds = controls + crit_var_set
    
                # Dependent variable
                form = '%s ~ 1' % dv
    
                # Fixed effects
                for pred in preds:
                    for S in range(N_SPILL):
                        if S < 1 and pred.startswith('incorrect'):  # Incorrect responses are filtered out, predictors have no variance
                            continue
                        if S < 2 and pred.startswith('endof'):  # Ends/starts are filtered out, predictors have no variance
                            continue
                        if S > 0 and pred in ('sentid', 'sentpos'):  # Trend predictors don't need to be spilled over
                            continue
                        _pred = []
                        for __pred in pred.split(':'):
                            if S > 0:
                                __pred = '%s_S%s' % (__pred, S)
                            __pred = __pred.replace('__', '_')
                            __pred = 'scale(%s)' % __pred
                            _pred.append(__pred)
                        form += ' + %s' % ':'.join(_pred)
    
                # Random slopes
                form += ' + (1'
                random_slopes = controls + crit_var_set
                for pred in random_slopes:
                    for S in range(N_SPILL):
                        if S < 1 and pred.startswith('incorrect'):  # Incorrect responses are filtered out, predictors have no variance
                            continue
                        if S < 2 and pred.startswith('endof'):  # Ends/starts are filtered out, predictors have no variance
                            continue
                        if S > 0 and pred in ('sentid', 'sentpos'):  # Trend predictors don't need to be spilled over
                            continue
                        _pred = []
                        for __pred in pred.split(':'):
                            if S > 0:
                                __pred = '%s_S%s' % (__pred, S)
                            __pred = __pred.replace('__', '_')
                            __pred = 'scale(%s)' % __pred
                            _pred.append(__pred)
                        form += ' + %s' % ':'.join(_pred)
                if len(random_slopes):
                    form += ' || subject)'
                else:
                    form += ' | subject)'
                
                # Random intercepts
                form += ' + (1 | docid_sentid_sentpos)'
                # form += ' + (1 | sentid)'
    
                out = R.format(
                    results_path=results_path,
                    corpus=corpus,
                    form=form,
                    dv=dv,
                    crit_var_set_name=crit_var_set_name
                )
    
                job_name = 'lme_%s_%s_%s_%s_fit' % (corpus, dv,  crit_var_set_name, eval_type)
                out_path = 'lme_scripts/%s.R' % job_name
                if not os.path.exists('lme_scripts'):
                    os.makedirs('lme_scripts')
    
                with open(out_path, 'w') as f:
                    f.write(out)
                os.chmod(out_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
    
                out_path = 'lme_scripts/%s.pbs' % job_name
                if corpus == 'geco':
                    mem = 16
                else:
                    mem = 16
                with open(out_path, 'w') as f:
                    f.write(bash_fit.format(job_name=job_name, mem=mem))
                os.chmod(out_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)

            # TESTING
            for comparison in comparisons:
                m1, m2 = comparison.split('_v_')

                # In-sample
                job_name = 'lme_%s_%s_%s_%s_test' % (corpus, dv, comparison, eval_type)
                if eval_type == 'insample':
                    out = R_test.format(
                        results_path=results_path,
                        corpus=corpus,
                        dv=dv,
                        m1=m1,
                        m2=m2
                    )
        
                    out_path = 'lme_scripts/%s.R' % job_name
                    with open(out_path, 'w') as f:
                        f.write(out)
                    os.chmod(out_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
    
                out_path = 'lme_scripts/%s.pbs' % job_name
                mem = 2
                with open(out_path, 'w') as f:
                    if eval_type == 'outofsample':
                        _dataset = corpus
                        _dv = dv
                        _comparison = comparison
                    else:
                        _dataset = _dv = _comparison = ''
                    f.write(bash_test.format(
                        job_name=job_name,
                        mem=mem,
                        dataset=_dataset,
                        dv=_dv,
                        comparison=_comparison
                    ))
                os.chmod(out_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
    

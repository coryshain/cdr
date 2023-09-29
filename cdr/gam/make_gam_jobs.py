import os, stat

R = '''#!/usr/bin/env Rscript

library(mgcv)

if (!dir.exists('{results_path}/{corpus}')) {{
  dir.create('{results_path}/{corpus}', recursive=TRUE)
}}

# Data loading
df_train = read.table("gam_data/{corpus}_X_train-dev.csv", sep=" ", header=TRUE)
df_train = df_train[is.finite(df_train${dv}),]
df_test = read.table("gam_data/{corpus}_X_test.csv", sep=" ", header=TRUE)
df_test = df_test[is.finite(df_test${dv}),]
for (col in c('subject', 'docid_sentid_sentpos')) {{
    df_train[[col]] = as.factor(df_train[[col]])
    df_test[[col]] = as.factor(df_test[[col]])
}}

# Fitting
form = {form}
m = bam(form, data=df_train, drop.unused.levels=FALSE, nthreads=4) 
model_path = '{results_path}/{corpus}/{dv}_{fn}.Rdata'
save(m, file=model_path)

# Plotting
plot_df = NULL
## Smooths
pd = plot(m)
N = 100
for (variable in pd) {{
    var_name = variable$xlab
    if (var_name != 'Gaussian quantiles') {{
        plot_df_ = data.frame(
           x = variable$x,
           y = variable$fit,
           err = variable$se
        )
        names(plot_df_) = c(paste0(var_name, '_x'), paste0(var_name, '_y'), paste0(var_name, '_err'))
        if (is.null(plot_df)) {{
            plot_df = plot_df_
        }} else {{
            plot_df = cbind(plot_df, plot_df_)
        }}
    }}
}}
## Linear terms
pd = termplot(m, se=TRUE, plot=FALSE)
n = nrow(plot_df)
for (var_name in names(pd)) {{
    # plot_df_ = pd[[var_name]]
    plot_data = pd[[var_name]]
    plot_df_ = NULL
    for (col_name in names(plot_data)) {{
        col = plot_data[[col_name]]
        start = col[1]
        end = col[length(col)]
        val = seq(start, end, length.out=n)
        if (is.null(plot_df_)) {{
            plot_df_ = data.frame(val)
            names(plot_df_) = c(col_name)
        }} else {{
            plot_df_[[col_name]] = val
        }}
    }}
    names(plot_df_) = c(paste0(var_name, '_x'), paste0(var_name, '_y'), paste0(var_name, '_err'))
    if (is.null(plot_df)) {{
        plot_df = plot_df_
    }} else {{
        plot_df = cbind(plot_df, plot_df_)
    }}
}}
plot_data_path = '{results_path}/{corpus}/{dv}_{fn}_plot.csv'
write.table(plot_df, plot_data_path, row.names=FALSE, col.names=TRUE, sep=',') 

# Evaluation
out = predict(m, newdata=df_test)
obs = df_test[['{dv}']]
err = obs - out
sqerr = err^2
mse = mean(sqerr)
variance = mean(residuals(m, type='response')^2)
ll = dnorm(err, mean=0, sd=sqrt(variance), log=TRUE)
ll_summed = sum(ll)
eval_str = paste0(
    '==================================================\\n',
    'GAM regression\\n\\n',
    'Model path: ', model_path, '\\n',

    'MODEL EVALUATION STATISTICS:\\n',
    '  Loglik: ', ll_summed, '\\n',
    '  MSE: ', mse, '\\n',
    '==================================================\\n'
)

cat(eval_str, file=stderr())

filename = '{results_path}/{corpus}/{dv}_{fn}_eval_test.txt'
write.table(eval_str, filename, row.names=FALSE, col.names=FALSE)

output = data.frame(GAMobs=obs, GAMpred=out, GAMerr=err, GAMloglik=ll)
filename = '{results_path}/{corpus}/{dv}_{fn}_output_test.csv'
write.table(output, filename, row.names=FALSE, col.names=TRUE, sep=',')
'''

bash = '''#!/bin/bash
#
#SBATCH --job-name={job_name}
#SBATCH --output="{job_name}-%N-%j.out"
#SBATCH --time=48:00:00
#SBATCH --mem={mem}gb
#SBATCH --ntasks=4

gam_scripts/{job_name}.R
'''

baseline_preds = {
    'brown': ['endofsentence', 'wlen', 'unigramsurp'],
    'dundee': ['inregression', 'endofsentence', 'endofline', 'endofscreen', 'wdelta', 'wlen', 'unigramsurp'],
    'geco': ['endofsentence', 'wdelta', 'wlen', 'unigramsurp'],
    'natstor': ['endofsentence', 'wlen', 'unigramsurp'],
    'natstormaze': ['incorrect', 'endofsentence', 'wlen', 'unigramsurp'],
    'provo': ['inregression', 'endofsentence', 'wdelta', 'wlen', 'unigramsurp']
}
dvs = {
    'brown': ['fdur'],
    'dundee': ['fdurSPsummed', 'fdurFP', 'fdurGP'],
    'geco': ['fdurFP', 'fdurGP'],
    'natstor': ['fdur'],
    'natstormaze': ['rt'],
    'provo': ['fdurSPsummed', 'fdurFP', 'fdurGP']
}

lms = ['nosurp', 'ngram', 'totsurp', 'gpt', 'gptj', 'gpt3', 'cloze']
fns = ['%s', '%sprob', 'pow0_5_%s_', 'pow0_75_%s_', 'pow1_%s_', 'pow1_33_%s_', 'pow2_%s_']

N_SPILL = 3

if os.path.exists('gam_results_path.txt'):
    with open('gam_results_path.txt') as f:
        for line in f:
           line = line.strip()
           if line:
               results_path = line
               break
else:
    results_path = 'results/gam'

for corpus in baseline_preds:
    dataset_basepath = 'gam_data/%s' % corpus
    for dv in dvs[corpus]:
        for lm in lms:
            if lm != 'cloze' or corpus == 'provo':
                for fn in fns:
                    preds = baseline_preds[corpus].copy()
                    if dv != 'fdurSPsummed':
                        preds = [x for x in preds if x != 'inregression']
                    if lm != 'nosurp':  # Add predictability variable
                        pred = fn % lm
                        if lm == 'cloze' and pred != 'clozeprob':
                            pred = pred.replace('cloze', 'clozesurp')
                        preds.append(pred)
    
                    # Dependent variable
                    form = '%s ~ 1' % dv
        
                    # Fixed effects
                    for pred in preds:
                        for S in range(N_SPILL):
                            if S < 1 and pred.startswith('incorrect'):  # Incorrect responses are filtered out, predictors have no variance
                                continue
                            if S < 2 and pred.startswith('endof'):  # Ends/starts are filtered out, predictors have no variance
                                continue
                            _pred = pred
                            if _pred.startswith('pow1_') and not _pred.startswith('pow1_33'):
                                _pred = _pred[5:-1]
                            if S > 0:
                                _pred = '%s_S%s' % (_pred, S)
                            _pred = _pred.replace('__', '_')
                            linear = pred.startswith('endof') or pred.startswith('inregression') or pred.startswith('incorrect') or 'prob' in pred or 'pow' in pred
                            if not linear:
                                _pred = 's(%s)' % _pred
                            form += ' + %s' % _pred
    
                    # Random slopes
                    for pred in preds:
                        for S in range(N_SPILL):
                            if S < 1 and pred.startswith('incorrect'):  # Incorrect responses are filtered out, predictors have no variance
                                continue
                            if S < 2 and pred.startswith('endof'):  # Ends/starts are filtered out, predictors have no variance
                                continue
                            _pred = pred
                            if _pred.startswith('pow1_') and not _pred.startswith('pow1_33'):
                                _pred = _pred[5:-1]
                            if S > 0:
                                _pred = '%s_S%s' % (_pred, S)
                            _pred = _pred.replace('__', '_')
                            form += ' + s(%s, subject, bs="re")' % _pred
                    
                    # Random intercepts
                    form += ' + s(subject, bs="re")'  # + s(docid_sentid_sentpos, bs="re")'
    
                    out = R.format(
                        results_path=results_path,
                        corpus=corpus,
                        form=form,
                        dv=dv,
                        fn='%s' % (fn % lm)
                    )
    
                    job_name = 'gam_%s_%s_%s' % (corpus, dv, fn % lm)
                    out_path = 'gam_scripts/%s.R' % job_name
                    if not os.path.exists('gam_scripts'):
                        os.makedirs('gam_scripts')
    
                    with open(out_path, 'w') as f:
                        f.write(out)
                    os.chmod(out_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
                    out_path = 'gam_scripts/%s.pbs' % job_name
                    if corpus == 'geco':
                        mem = 16
                    else:
                        mem = 16
                    with open(out_path, 'w') as f:
                        f.write(bash.format(job_name=job_name, mem=mem))
                    os.chmod(out_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
    
                    if lm == 'nosurp':
                        break

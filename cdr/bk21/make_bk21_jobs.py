import os, stat


def get_R_code(model_type):
    if model_type.lower() == 'gam':
       model_string = R_GAM_model
       predict_string = R_GAM_predict
       plot_string = R_GAM_plot
    elif model_type.lower() == 'lme':
       model_string = R_LME_model
       predict_string = R_LME_predict
       plot_string = R_LME_plot
    return R_preamble + R_data + model_string + plot_string + predict_string + R_evaluation


def check_config(name):
    if name.startswith('nosurp'):
        if 'prob' in name:
            return False
        return True
    return True

def is_predictability(name):
    for lm in lms:
        if name.startswith(lm):
            return True
    return False

def make_job(name, experiment, control, is_gam=True, results_path='results/bk21'):
    ranefs = ranef_by_experiment[experiment]
    dv = dv_by_experiment[experiment]
    model = 'gam' if is_gam else 'lme'
    if check_config(name):
        if control:
            preds = ['critical_word_pos', 'wlen', 'wlenregion', 'unigram', 'unigramregion', 'glovedistmean']
        else:
            preds = []
        if name != 'nosurp':
            if '-' in name:
                preds += name.split('-')
            else:
                preds.append(name)
        form = '%s ~ 1' % dv
        if is_gam:
            for pred in preds:
                if pred == 'critical_word_pos':
                    if experiment == 'naming':
                        k = ', k=8'  # 8 unique values
                    else:  # experiment == 'spr'
                        k = ', k=9'  # 9 unique values
                elif pred == 'wlen':
                        k = ', k=6'  # 6 unique values
                else:
                    k = ''
                term = ' + s(%s, bs="cs"%s)' % (pred, k)
                form += term
            random = []
            for ranef in ranefs:
                random.append('s(%s, bs="re")' % ranef)
                ## Random slopes cause convergence erros, removed
                #if ranef.lower() != 'item':
                #    for pred in preds:
                #        if is_predictability(pred):
                #            random.append('s(%s, %s, bs="re")' % (pred, ranef))
            random = ' + '.join(random)
            form += ' + ' + random
        else:
            for pred in preds:
                term = ' + %s' % pred
                form += term
            random = []
            for ranef in ranefs:
                ranef_terms = ['1']
                ## Random slopes cause convergence erros, removed
                #if ranef != 'itemnum':
                #    for pred in preds:
                #        if is_predictability(pred):
                #            ranef_terms.append(pred)
                random.append('(%s | %s)' % (' + '.join(ranef_terms), ranef))
            random = ' + '.join(random)
            form += ' + ' + random
    
        job_name = '%s_%s_%s%s' % (model, experiment, name, control)                

        out = get_R_code(model).format(
            results_path=results_path,
            experiment=experiment,
            form=form,
            dv=dv,
            model=job_name
        )

        out_path = 'bk21_scripts/%s.R' % job_name
        if not os.path.exists('bk21_scripts'):
            os.makedirs('bk21_scripts')

        with open(out_path, 'w') as f:
            f.write(out)
        os.chmod(out_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
        out_path = 'bk21_scripts/%s.pbs' % job_name
        mem = 16
        with open(out_path, 'w') as f:
            f.write(bash.format(job_name=job_name, mem=mem))
        os.chmod(out_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)

R_preamble = '''#!/usr/bin/env Rscript

NFOLDS = 5

pr = function(x, buffer=NULL) {{
    if (is.null(buffer)) {{
        buffer = stderr()
    }}
    cat(paste0(x, '\\n'), file=buffer, append=TRUE)
}}

if (!dir.exists('{results_path}')) {{
  dir.create('{results_path}', recursive=TRUE)
}}

'''

R_data = '''# Data
pr('Loading data')
df = read.csv("bk21_data/bk21_{experiment}.csv", header=TRUE)
for (col in c('SUB', 'subject', 'ITEM', 'item', 'itemnum')) {{
    if (col %in% colnames(df)) {{
        df[[col]] = as.factor(df[[col]])
    }}
}}
df = df[is.finite(df${dv}),]

'''

R_GAM_model = '''# Fitting
pr('Fitting')
library(mgcv)
m = gam({form}, data=df)

# Summary
summary_path = '{results_path}/{model}.summary.txt'
sink(summary_path)
print(summary(m))
pr(paste0('\\nLogLik: ', logLik(m)), stdout())
pr(paste0('AIC: ', extractAIC(m)[[2]], '\\n'), stdout())
sink()

'''

R_GAM_predict = '''# Prediction
pr('Predicting')
out = predict(m, newdata=df)
sd = sqrt(mean(residuals(m, type='response')^2))
sd = rep(sd, length(out))

'''

R_GAM_plot = '''# Plotting
pr('Plotting')
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
plot_data_path = '{results_path}/{model}_plot.csv'
write.table(plot_df, plot_data_path, row.names=FALSE, col.names=TRUE, sep=',') 

'''

R_LME_model = '''# Fitting
pr('Fitting')
library(lme4)
m = list()
for (fold in 1:NFOLDS) {{
    df_ = df[df$fold != fold,]
    pr(paste0('Fold ', fold, ' size: ', nrow(df) - nrow(df_)))
    m_ = lmer({form}, data=df_, REML=FALSE)
    m[[fold]] = m_
}}

'''

R_LME_plot = ''

R_LME_predict = '''# Prediction
pr('Predicting')
out = list()
for (fold in 1:NFOLDS) {{
    df_ = df[df$fold == fold,]
    out_ = predict(m[[fold]], newdata=df_, allow.new.levels=TRUE)
    sd = sqrt(mean(residuals(m[[fold]], type='response')^2))
    sd = rep(sd, length(out_))
    out_ = data.frame(list(pred=out_, sd=sd, sortix=df_$sortix))
    out[[fold]] = out_
}}
out = do.call(rbind, out)
out = out[order(out$sortix),]
sd = out$sd
out = out$pred

'''

R_evaluation = '''# Evaluation
pr('Evaluating')
obs = df[['{dv}']]
err = obs - out
sqerr = err^2
mse = mean(sqerr)
ll = dnorm(err, mean=0, sd=sd, log=TRUE)
ll_summed = sum(ll)
eval_str = paste0(
    '==================================================\\n',
    'Model name: {model}\\n',

    'MODEL EVALUATION STATISTICS:\\n',
    '  Loglik: ', ll_summed, '\\n',
    '  MSE: ', mse, '\\n',
    '==================================================\\n'
)

cat(eval_str, file=stderr())

filename = '{results_path}/{model}_eval_test.txt'
write.table(eval_str, filename, row.names=FALSE, col.names=FALSE)

output = data.frame(obs=obs, pred=out, err=err, loglik=ll)
filename = '{results_path}/{model}_output_test.csv'
write.table(output, filename, row.names=FALSE, col.names=TRUE, sep=',')

'''

bash = '''#!/bin/bash
#
#SBATCH --job-name={job_name}
#SBATCH --output="{job_name}-%N-%j.out"
#SBATCH --time=4:00:00
#SBATCH --mem={mem}gb
#SBATCH --ntasks=1

bk21_scripts/{job_name}.R
'''

experiments = ['spr', 'naming']
dv_by_experiment = {'spr': 'SUM_3RT_trimmed', 'naming': 'TRIM_RT'}
ranef_by_experiment = {'spr': ('SUB', 'ITEM'), 'naming': ('subject', 'item')}
fns = ['%s', '%sprob']
controls = ['', 'C']
if os.path.exists('bk21_results_path.txt'):
    with open('bk21_results_path.txt') as f:
        for line in f:
           line = line.strip()
           if line:
               results_path = line
               break
else:
    results_path = 'results/bk21'

# GAM models
lms = ['nosurp', 'cloze', 'trigram', 'gpt2', 'gpt2region']
for experiment in experiments:
    for lm in lms:
        for fn in fns:
            name = fn % lm
            for control in controls:
                make_job(name, experiment, control, is_gam=True)

# LME models
## Within LMs
fns = ['prob', 'surp', 'both']
for experiment in experiments:
    for lm in lms:
        for fn in fns:
            if fn == 'prob':
                name = '%sprob' % lm
            elif fn == 'surp':
                name = lm
            else:  # fn == 'both'
                name = '%sprob-%s' % (lm, lm)
            for control in controls:
                make_job(name, experiment, control, is_gam=False)

## Between LMs
pred_sets = [
    ['clozeprob', 'gpt2'],
    ['clozeprob', 'gpt2region'],
    ['clozeprob', 'trigram'],
]
for experiment in experiments:
    for pred_set in pred_sets:
        name = '-'.join(pred_set)
        for control in controls:
            make_job(name, experiment, control, is_gam=False, results_path=results_path)

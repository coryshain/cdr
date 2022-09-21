import os

template_R = '''#!/usr/bin/env Rscript

library(gamlss)


# Configurables

df_path = '%s'
out_dir = '%s/'
bform = '%s'
dv = '%s'
model_name = '%s'


# Main Code

df_train = read.csv(paste0(df_path, '_train.csv'), sep=' ', header=T)
df_dev = read.csv(paste0(df_path, '_dev.csv'), sep=' ', header=T)
df_test = read.csv(paste0(df_path, '_test.csv'), sep=' ', header=T)

df_train = df_train[, colSums(is.na(df_train))==0]
df_dev = df_dev[, colSums(is.na(df_dev))==0]
df_test = df_test[, colSums(is.na(df_test))==0]

df_train$subject = as.factor(df_train$subject)
df_dev$subject = as.factor(df_dev$subject)
df_test$subject = as.factor(df_test$subject)

if ('fROI' %%in%% colnames(df_train)) {
    df_train$fROI = as.factor(df_train$fROI)
}
if ('fROI' %%in%% colnames(df_dev)) {
    df_dev$fROI = as.factor(df_dev$fROI)
}
if ('fROI' %%in%% colnames(df_test)) {
    df_test$fROI = as.factor(df_test$fROI)
}

dir.create(out_dir, recursive=TRUE)
out_path = paste0(out_dir, 'm.Rdata')

bform_mu = as.formula(paste0(dv, ' ~ ', bform))
bform_sigma = as.formula(paste0('~ ', bform))

if (file.exists(out_path)) {
    cat('Loading saved model...\\n', file=stderr())
    m = get(load(out_path))
} else {
    cat('Regressing GAMLSS model...\\n', file=stderr())
    m = gamlss(bform_mu, sigma.formula=as.formula(bform_sigma), data=df_train)
}

save(m, file=out_path)

p_df = list(df_train, df_dev, df_test)
p_names = c('train', 'dev', 'test')

cat('Evaluating...\\n', file=stderr())

for (i in 1:3) {
    p_df_ = p_df[[i]]
    p_name = p_names[i]
    for (param in c('mu', 'sigma')) {
        out = predict(m, what=param, type='response', newdata=p_df_)
        if (param == 'mu') {
            obs = p_df_[[dv]]
            err = obs - out
            sqerr = err^2
            mse = mean(sqerr)
            mae = mean(abs(err))
            eval_str = paste0(
                '==================================================\\n',
                'GAMLSS regression\\n\\n',
                'Model name: ', model_name, '\\n',
                'Loss (', p_name, ' set):\\n',
                '  MSE: ', mse, '\\n',
                '  MAE: ', mae, '\\n',
                '==================================================\\n'
            )
            filename = paste0('eval_', p_name, '.txt')
            write.table(eval_str, paste0(out_dir, filename), row.names=FALSE, col.names=FALSE)
            
            filename = paste0('obs_', p_name, '.txt')
            write.table(obs, paste0(out_dir, filename), row.names=FALSE, col.names=FALSE)
            
            filename = paste0('preds_', p_name, '.txt')
            write.table(out, paste0(out_dir, filename), row.names=FALSE, col.names=FALSE)
            
            filename = paste0('losses_mse_', p_name, '.txt')
            write.table(sqerr, paste0(out_dir, filename), row.names=FALSE, col.names=FALSE)
        } else {
            filename = paste0('preds_sigma_', p_name, '.txt')
            write.table(out, paste0(out_dir, filename), row.names=FALSE, col.names=FALSE)
        }
    }
}

'''

template_SLURM = '''#!/bin/bash
#
#SBATCH --job-name=%s
#SBATCH --output="%s-%%N-%%j.out"
#SBATCH --time=72:00:00
#SBATCH --mem=64gb
#SBATCH --ntasks=10
#SBATCH --partition=evlab

gamlss_scripts/%s.R
'''

names = [
    'dundee_%s_sp',
    'dundee_%s_fp',
    'dundee_%s_gp',
    'natstor_%s',
    'fmri%s'
]

dvs = [
    'fdurSPsummed',
    'fdurFP',
    'fdurGP',
    'fdur',
    'BOLD'
]

dfs = [
    'baseline_data/dundee_SP_X',
    'baseline_data/dundee_FPGP_X',
    'baseline_data/dundee_FPGP_X_2',
    'baseline_data/natstor_X',
    'baseline_data/LANG_convolved_X'
]

dv_prefixes = [
    '',
    'log_'
]

bforms = {
    'dundee_%s_sp': {
        'noS': 'pb(z_trial) + pb(z_sentpos) + notregression + z_wdeltanotreg + z_wdeltareg + prevwasfixnotreg + prevwasfixreg + pb(z_wlennotreg) + pb(z_wlenreg) + pb(z_unigramsurpnotreg) + pb(z_unigramsurpreg) + pb(z_fwprob5surpnotreg) + pb(z_fwprob5surpreg) + random(subject)',
        'fullS': 'pb(z_trial) + pb(z_sentpos) + notregression + notregressionS1 + notregressionS2 + notregressionS3 + z_wdeltanotreg +  z_wdeltareg + z_wdeltanotregS1 + z_wdeltaregS1 + z_wdeltanotregS2 + z_wdeltaregS2 + z_wdeltanotregS3 + z_wdeltaregS3 + prevwasfixnotreg + prevwasfixreg + prevwasfixnotregS1 + prevwasfixregS1 + prevwasfixnotregS2 + prevwasfixregS2 + prevwasfixnotregS3 + prevwasfixregS3 + pb(z_wlennotreg) + pb(z_wlenreg) + pb(z_wlennotregS1) + pb(z_wlenregS1) + pb(z_wlennotregS2) + pb(z_wlenregS2) + pb(z_wlennotregS3) + pb(z_wlenregS3) + pb(z_unigramsurpnotreg) + pb(z_unigramsurpreg) + pb(z_unigramsurpnotregS1) + pb(z_unigramsurpregS1) + pb(z_unigramsurpnotregS2) + pb(z_unigramsurpregS2) + pb(z_unigramsurpnotregS3) + pb(z_unigramsurpregS3) + pb(z_fwprob5surpnotreg) + pb(z_fwprob5surpreg) + pb(z_fwprob5surpnotregS1) + pb(z_fwprob5surpregS1) + pb(z_fwprob5surpnotregS2) + pb(z_fwprob5surpregS2) + pb(z_fwprob5surpnotregS3) + pb(z_fwprob5surpregS3) + random(subject)'
    },
    'dundee_%s_fp': {
        'noS': 'pb(z_trial) + pb(z_sentpos) + z_wdelta + prevwasfix + pb(z_wlen) + pb(z_unigramsurp) + pb(z_fwprob5surp) + random(subject)',
        'fullS': 'pb(z_trial) + pb(z_sentpos) + z_wdelta + z_wdeltaS1 + z_wdeltaS2 + z_wdeltaS3 + prevwasfix + prevwasfixS1 + prevwasfixS2 + prevwasfixS3 + pb(z_wlen) + pb(z_wlenS1) + pb(z_wlenS2) + pb(z_wlenS3) + pb(z_unigramsurp) + pb(z_unigramsurpS1) + pb(z_unigramsurpS2) + pb(z_unigramsurpS3) + pb(z_fwprob5surp) + pb(z_fwprob5surpS1) + pb(z_fwprob5surpS2) + pb(z_fwprob5surpS3) + random(subject)'
    },
    'dundee_%s_gp': {
        'noS': 'pb(z_trial) + pb(z_sentpos) + z_wdelta + prevwasfix + pb(z_wlen) + pb(z_unigramsurp) + pb(z_fwprob5surp) + random(subject)',
        'fullS': 'pb(z_trial) + pb(z_sentpos) + z_wdelta + z_wdeltaS1 + z_wdeltaS2 + z_wdeltaS3 + prevwasfix + prevwasfixS1 + prevwasfixS2 + prevwasfixS3 + pb(z_wlen) + pb(z_wlenS1) + pb(z_wlenS2) + pb(z_wlenS3) + pb(z_unigramsurp) + pb(z_unigramsurpS1) + pb(z_unigramsurpS2) + pb(z_unigramsurpS3) + pb(z_fwprob5surp) + pb(z_fwprob5surpS1) + pb(z_fwprob5surpS2) + pb(z_fwprob5surpS3) + random(subject)'
    },
    'natstor_%s': {
        'noS': 'pb(z_trial) + pb(z_sentpos) + pb(z_wlen) + pb(z_unigramsurp) + pb(z_fwprob5surp) + re(random=~1|subject) + random(subject)',
        'fullS': 'pb(z_trial) + pb(z_sentpos) + pb(z_wlen) + pb(z_wlenS1) + pb(z_wlenS2) + pb(z_wlenS3) + pb(z_unigramsurp) + pb(z_unigramsurpS1) + pb(z_unigramsurpS2) + pb(z_unigramsurpS3) + pb(z_fwprob5surp) + pb(z_fwprob5surpS1) + pb(z_fwprob5surpS2) + pb(z_fwprob5surpS3) + random(subject)'
    },
    'fmri%s': {
        'noS': 'pb(z_tr) + pb(z_Rate) + pb(z_soundPower100ms) + pb(z_unigramsurp) + pb(fwprob5surp) + random(subject) + random(fROI)'
    }
    
}

bforms_re = {
    'dundee_%s_sp': {
        'noS': 'pb(z_trial) + pb(z_sentpos) + notregression + z_wdeltanotreg + z_wdeltareg + prevwasfixnotreg + prevwasfixreg + pb(z_wlennotreg) + pb(z_wlenreg) + pb(z_unigramsurpnotreg) + pb(z_unigramsurpreg) + pb(z_fwprob5surpnotreg) + pb(z_fwprob5surpreg) + re(random=~1|subject) + re(random=~z_trial|subject) + re(random=~z_sentpos|subject) + re(random=~notregression|subject) + re(random=~z_wdeltanotreg|subject) + re(random=~z_wdeltareg|subject) + re(random=~prevwasfixnotreg|subject) + re(random=~prevwasfixreg|subject) + re(random=~z_wlennotreg|subject) + re(random=~z_wlenreg|subject) + re(random=~z_unigramsurpnotreg|subject) + re(random=~z_unigramsurpreg|subject) + re(random=~z_fwprob5surpnotreg|subject) + re(random=~z_fwprob5surpreg|subject)',
        'fullS': 'pb(z_trial) + pb(z_sentpos) + notregression + notregressionS1 + notregressionS2 + notregressionS3 + z_wdeltanotreg +  z_wdeltareg + z_wdeltanotregS1 + z_wdeltaregS1 + z_wdeltanotregS2 + z_wdeltaregS2 + z_wdeltanotregS3 + z_wdeltaregS3 + prevwasfixnotreg + prevwasfixreg + prevwasfixnotregS1 + prevwasfixregS1 + prevwasfixnotregS2 + prevwasfixregS2 + prevwasfixnotregS3 + prevwasfixregS3 + pb(z_wlennotreg) + pb(z_wlenreg) + pb(z_wlennotregS1) + pb(z_wlenregS1) + pb(z_wlennotregS2) + pb(z_wlenregS2) + pb(z_wlennotregS3) + pb(z_wlenregS3) + pb(z_unigramsurpnotreg) + pb(z_unigramsurpreg) + pb(z_unigramsurpnotregS1) + pb(z_unigramsurpregS1) + pb(z_unigramsurpnotregS2) + pb(z_unigramsurpregS2) + pb(z_unigramsurpnotregS3) + pb(z_unigramsurpregS3) + pb(z_fwprob5surpnotreg) + pb(z_fwprob5surpreg) + pb(z_fwprob5surpnotregS1) + pb(z_fwprob5surpregS1) + pb(z_fwprob5surpnotregS2) + pb(z_fwprob5surpregS2) + pb(z_fwprob5surpnotregS3) + pb(z_fwprob5surpregS3) + re(random=~1|subject) + re(random=~z_trial|subject) + re(random=~z_sentpos|subject) + re(random=~notregression|subject) + re(random=~notregressionS1|subject) + re(random=~notregressionS2|subject) + re(random=~notregressionS3|subject) + re(random=~z_wdeltanotreg|subject) + re(random=~z_wdeltareg|subject) + re(random=~z_wdeltanotregS1|subject) + re(random=~z_wdeltaregS1|subject) + re(random=~z_wdeltanotregS2|subject) + re(random=~z_wdeltaregS2|subject) + re(random=~z_wdeltanotregS3|subject) + re(random=~z_wdeltaregS3|subject) + re(random=~prevwasfixnotreg|subject) + re(random=~prevwasfixreg|subject) + re(random=~prevwasfixnotregS1|subject) + re(random=~prevwasfixregS1|subject) + re(random=~prevwasfixnotregS2|subject) + re(random=~prevwasfixregS2|subject) + re(random=~prevwasfixnotregS3|subject) + re(random=~prevwasfixregS3|subject) + re(random=~z_wlennotreg|subject) + re(random=~z_wlenreg|subject) + re(random=~z_wlennotregS1|subject) + re(random=~z_wlenregS1|subject) + re(random=~z_wlennotregS2|subject) + re(random=~z_wlenregS2|subject) + re(random=~z_wlennotregS3|subject) + re(random=~z_wlenregS3|subject) + re(random=~z_unigramsurpnotreg|subject) + re(random=~z_unigramsurpreg|subject) + re(random=~z_unigramsurpnotregS1|subject) + re(random=~z_unigramsurpregS1|subject) + re(random=~z_unigramsurpnotregS2|subject) + re(random=~z_unigramsurpregS2|subject) + re(random=~z_unigramsurpnotregS3|subject) + re(random=~z_unigramsurpregS3|subject) + re(random=~z_fwprob5surpnotreg|subject) + re(random=~z_fwprob5surpreg|subject) + re(random=~z_fwprob5surpnotregS1|subject) + re(random=~z_fwprob5surpregS1|subject) + re(random=~z_fwprob5surpnotregS2|subject) + re(random=~z_fwprob5surpregS2|subject) + re(random=~z_fwprob5surpnotregS3|subject) + re(random=~z_fwprob5surpregS3|subject)'
    },
    'dundee_%s_fp': {
        'noS': 'pb(z_trial) + pb(z_sentpos) + z_wdelta + prevwasfix + pb(z_wlen) + pb(z_unigramsurp) + pb(z_fwprob5surp) + re(random=~1|subject) + re(random=~z_trial|subject) + re(random=~z_sentpos|subject) + re(random=~z_wdelta|subject) + re(random=~prevwasfix|subject) + re(random=~z_wlen|subject) + re(random=~z_unigramsurp|subject) + re(random=~z_fwprob5surp|subject)',
        'fullS': 'pb(z_trial) + pb(z_sentpos) + z_wdelta + z_wdeltaS1 + z_wdeltaS2 + z_wdeltaS3 + prevwasfix + prevwasfixS1 + prevwasfixS2 + prevwasfixS3 + pb(z_wlen) + pb(z_wlenS1) + pb(z_wlenS2) + pb(z_wlenS3) + pb(z_unigramsurp) + pb(z_unigramsurpS1) + pb(z_unigramsurpS2) + pb(z_unigramsurpS3) + pb(z_fwprob5surp) + pb(z_fwprob5surpS1) + pb(z_fwprob5surpS2) + pb(z_fwprob5surpS3) + re(random=~1|subject) + re(random=~z_trial|subject) + re(random=~z_sentpos|subject) + re(random=~z_wdelta|subject) + re(random=~z_wdeltaS1|subject) + re(random=~z_wdeltaS2|subject) + re(random=~z_wdeltaS3|subject) + re(random=~prevwasfix|subject) + re(random=~prevwasfixS1|subject) + re(random=~prevwasfixS2|subject) + re(random=~prevwasfixS3|subject) + re(random=~z_wlen|subject) + re(random=~z_wlenS1|subject) + re(random=~z_wlenS2|subject) + re(random=~z_wlenS3|subject) + re(random=~z_unigramsurp|subject) + re(random=~z_unigramsurpS1|subject) + re(random=~z_unigramsurpS2|subject) + re(random=~z_unigramsurpS3|subject) + re(random=~z_fwprob5surp|subject) + re(random=~z_fwprob5surpS1|subject) + re(random=~z_fwprob5surpS2|subject) + re(random=~z_fwprob5surpS3|subject)'
    },
    'dundee_%s_gp': {
        'noS': 'pb(z_trial) + pb(z_sentpos) + z_wdelta + prevwasfix + pb(z_wlen) + pb(z_unigramsurp) + pb(z_fwprob5surp) + re(random=~1|subject) + re(random=~z_trial|subject) + re(random=~z_sentpos|subject) + re(random=~z_wdelta|subject) + re(random=~prevwasfix|subject) + re(random=~z_wlen|subject) + re(random=~z_unigramsurp|subject) + re(random=~z_fwprob5surp|subject)',
        'fullS': 'pb(z_trial) + pb(z_sentpos) + z_wdelta + z_wdeltaS1 + z_wdeltaS2 + z_wdeltaS3 + prevwasfix + prevwasfixS1 + prevwasfixS2 + prevwasfixS3 + pb(z_wlen) + pb(z_wlenS1) + pb(z_wlenS2) + pb(z_wlenS3) + pb(z_unigramsurp) + pb(z_unigramsurpS1) + pb(z_unigramsurpS2) + pb(z_unigramsurpS3) + pb(z_fwprob5surp) + pb(z_fwprob5surpS1) + pb(z_fwprob5surpS2) + pb(z_fwprob5surpS3) + re(random=~1|subject) + re(random=~z_trial|subject) + re(random=~z_sentpos|subject) + re(random=~z_wdelta|subject) + re(random=~z_wdeltaS1|subject) + re(random=~z_wdeltaS2|subject) + re(random=~z_wdeltaS3|subject) + re(random=~prevwasfix|subject) + re(random=~prevwasfixS1|subject) + re(random=~prevwasfixS2|subject) + re(random=~prevwasfixS3|subject) + re(random=~z_wlen|subject) + re(random=~z_wlenS1|subject) + re(random=~z_wlenS2|subject) + re(random=~z_wlenS3|subject) + re(random=~z_unigramsurp|subject) + re(random=~z_unigramsurpS1|subject) + re(random=~z_unigramsurpS2|subject) + re(random=~z_unigramsurpS3|subject) + re(random=~z_fwprob5surp|subject) + re(random=~z_fwprob5surpS1|subject) + re(random=~z_fwprob5surpS2|subject) + re(random=~z_fwprob5surpS3|subject)'
    },
    'natstor_%s': {
        'noS': 'pb(z_trial) + pb(z_sentpos) + pb(z_wlen) + pb(z_unigramsurp) + pb(z_fwprob5surp) + re(random=~1|subject) + re(random=~z_trial|subject) + re(random=~z_sentpos|subject) + re(random=~z_wlen|subject) + re(random=~z_unigramsurp|subject) + re(random=~z_fwprob5surp|subject)',
        'fullS': 'pb(z_trial) + pb(z_sentpos) + pb(z_wlen) + pb(z_wlenS1) + pb(z_wlenS2) + pb(z_wlenS3) + pb(z_unigramsurp) + pb(z_unigramsurpS1) + pb(z_unigramsurpS2) + pb(z_unigramsurpS3) + pb(z_fwprob5surp) + pb(z_fwprob5surpS1) + pb(z_fwprob5surpS2) + pb(z_fwprob5surpS3) + re(random=~1|subject) + re(random=~z_trial|subject) + re(random=~z_sentpos|subject) + re(random=~z_wlen|subject) + re(random=~z_wlenS1|subject) + re(random=~z_wlenS2|subject) + re(random=~z_wlenS3|subject) + re(random=~z_unigramsurp|subject) + re(random=~z_unigramsurpS1|subject) + re(random=~z_unigramsurpS2|subject) + re(random=~z_unigramsurpS3|subject) + re(random=~z_fwprob5surp|subject) + re(random=~z_fwprob5surpS1|subject) + re(random=~z_fwprob5surpS2|subject) + re(random=~z_fwprob5surpS3|subject)'
    }
}

if not os.path.exists('gamlss_scripts'):
    os.makedirs('gamlss_scripts')

for name, dv, df in zip(names, dvs, dfs):
    for lag in ('noS', 'fullS'):
        for dv_type in ('raw', 'log'):
            if name != 'fmri%s' or (dv_type != 'log' and lag != 'fullS'):
                bform = bforms[name][lag]
                if dv_type == 'raw':
                    _dv = dv
                else:
                    _dv = 'log_' + dv
                model_name = '%s_GAMLSS_%s' % (name % dv_type, lag)
                out_dir = '../results/cdrnn_journal/%s/GAMLSS%s' % (name % dv_type, lag)
                script_R = template_R % (df, out_dir, bform, _dv, 'GAMLSS%s' % lag)
                with open('gamlss_scripts/%s.R' % model_name, 'w') as f:
                    f.write(script_R)
                script_SLURM = template_SLURM % (model_name, model_name, model_name)
                with open('gamlss_scripts/%s.pbs' % model_name, 'w') as f:
                    f.write(script_SLURM)

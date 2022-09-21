#!/usr/bin/env Rscript

library(mgcv)


# Configurables

df_path = 'baseline_data/LANG_convolved_X'
out_dir = '../results/cdrnn_journal/fmriraw/GAMnoS/'
bform = 's(z_tr) + s(z_Rate) + s(z_soundPower100ms) + s(z_unigramsurp) + s(fwprob5surp) + s(subject, bs="re") + s(fROI, bs="re") + s(z_tr, fROI, bs="re") + s(z_Rate, fROI, bs="re") + s(z_soundPower100ms, fROI, bs="re") + s(z_unigramsurp, fROI, bs="re") + s(z_fwprob5surp, fROI, bs="re")'
dv = 'BOLD'
model_name = 'GAMLSSnoS'


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

if ('fROI' %in% colnames(df_train)) {
    df_train$fROI = as.factor(df_train$fROI)
}
if ('fROI' %in% colnames(df_dev)) {
    df_dev$fROI = as.factor(df_dev$fROI)
}
if ('fROI' %in% colnames(df_test)) {
    df_test$fROI = as.factor(df_test$fROI)
}

dir.create(out_dir, recursive=TRUE)
out_path = paste0(out_dir, 'm.Rdata')

bform_mu = as.formula(paste0(dv, ' ~ ', bform))
bform_sigma = as.formula(paste0('~ ', bform))

if (file.exists(out_path)) {
    cat('Loading saved model...\n', file=stderr())
    m = get(load(out_path))
} else {
    cat('Regressing GAM model...\n', file=stderr())
    m = bam(bform_mu, data=df_train, drop.unused.levels=FALSE, nthreads=10)
}

save(m, file=out_path)

p_df = list(df_train, df_dev, df_test)
p_names = c('train', 'dev', 'test')

cat('Evaluating...\n', file=stderr())

for (i in 1:3) {
    p_df_ = p_df[[i]]
    p_name = p_names[i]
    for (param in c('mu')) {
        out = predict(m, newdata=p_df_, na.action=na.pass)
        if (param == 'mu') {
            obs = p_df_[[dv]]
            err = obs - out
            sqerr = err^2
            mse = mean(sqerr)
            mae = mean(abs(err))
            eval_str = paste0(
                '==================================================\n',
                'GAMLSS regression\n\n',
                'Model name: ', model_name, '\n',
                'Loss (', p_name, ' set):\n',
                '  MSE: ', mse, '\n',
                '  MAE: ', mae, '\n',
                '==================================================\n'
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


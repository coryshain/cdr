#!/usr/bin/env Rscript

pr = function(x, buffer=NULL) {
    if (is.null(buffer)) {
        buffer = stderr()
    }
    cat(x, file=buffer, append=TRUE)
}

path = 'data/SPR_brotherskuperberg/orig'
df = read.csv(file=paste0(path, 'AppendixB.csv'),head=T)
df$item = as.integer(as.factor(df$item))

df[['fold']] = (df$item %% 5) + 1
df[['sortix']] = 1:nrow(df)

for (dvnum in 1:5) {
    if (dvnum == 1) {
        model_name = 'Lexical Decision'
        dv = 'LDT'
    } else if (dvnum == 2) {
        model_name = 'Orthographic Neighborhood Size'
        dv = 'ORTHN'
    } else if (dvnum == 3) {
        model_name = 'Age of Acquisition'
        dv = 'AOA'
    } else if (dvnum == 4) {
        model_name = 'SUBTLEX Log Frequency'
        dv = 'SUBTLEX'
    } else if (dvnum == 5) {
        model_name = 'Word Length'
        dv = 'LENGTH'
    }

    df_ = df[is.finite(df[[dv]]),]

    # Standard regression
    form = as.formula(paste0(dv, '~ 1 + UNILOG + TRILOG'))
    m = lm(form, data=df_)
    
    pr('================================================\n', stdout())
    pr(paste0(model_name, ' Model:\n'), stdout())
    print(summary(m))
    pr('------------------------------------------------\n', stdout())
    
    # Cross-validated regression
    pred_no_uni = list()
    pred_no_tri = list()
    pred_full = list()
    for (fold in 1:5) {
        train = df_[df_$fold != fold,]
        test = df_[df_$fold == fold,]
    
        # No unigram
        form = as.formula(paste0(dv, '~ 1 + TRILOG'))
        m = lm(form, data=train)
        sd = sqrt(mean(residuals(m, type='response')^2))
        pred = predict(m, newdata=test)
        obs = test[[dv]]
        err = obs - pred
        ll = dnorm(err, mean=0, sd=sd, log=TRUE)
        pred_no_uni[[fold]] = data.frame(list(ll=ll, sortix=test$sortix))
    
        # No trigram
        form = as.formula(paste0(dv, '~ 1 + UNILOG'))
        m = lm(form, data=train)
        sd = sqrt(mean(residuals(m, type='response')^2))
        pred = predict(m, newdata=test)
        obs = test[[dv]]
        err = obs - pred
        ll = dnorm(err, mean=0, sd=sd, log=TRUE)
        pred_no_tri[[fold]] = data.frame(list(ll=ll, sortix=test$sortix))
    
        # Full
        form = as.formula(paste0(dv, '~ 1 + UNILOG + TRILOG'))
        m = lm(form, data=train)
        sd = sqrt(mean(residuals(m, type='response')^2))
        pred = predict(m, newdata=test)
        obs = test[[dv]]
        err = obs - pred
        ll = dnorm(err, mean=0, sd=sd, log=TRUE)
        pred_full[[fold]] = data.frame(list(ll=ll, sortix=test$sortix))
    }
    
    pred_no_uni = do.call(rbind, pred_no_uni)
    pred_no_uni = pred_no_uni[order(pred_no_uni$sortix),]
    pred_no_tri = do.call(rbind, pred_no_tri)
    pred_no_tri = pred_no_tri[order(pred_no_tri$sortix),]
    pred_full = do.call(rbind, pred_full)
    pred_full = pred_full[order(pred_full$sortix),]
    
    ll_no_uni = pred_no_uni$ll
    ll_no_tri = pred_no_tri$ll
    ll_full = pred_full$ll
    
    NSAMP = 10000
    diff_uni = abs(sum(ll_full) - sum(ll_no_uni))
    diff_tri = abs(sum(ll_full) - sum(ll_no_tri))
    
    pr('Testing unigram effects\n')
    diffs = numeric(NSAMP)
    for (i in 1:NSAMP) {
        pr(paste0('\r', i, '/', NSAMP))
        a = numeric(length(pred_no_uni))
        b = numeric(length(pred_full))
        samp = as.numeric(runif(NSAMP) > 0.5)
        a = ll_no_uni * samp + ll_full * (1 - samp)
        b = ll_no_uni * (1 - samp) + ll_full * samp
    
        diff = abs(sum(a) - sum(b))
        diffs[i] = diff
    }
    pr('\n\n')
    n_greater = sum(diffs > diff_uni)
    p_uni = (n_greater + 1) / (NSAMP + 1)
    
    pr('Testing trigram effects\n')
    diffs = numeric(NSAMP)
    for (i in 1:NSAMP) {
        pr(paste0('\r', i, '/', NSAMP))
        a = numeric(length(pred_no_tri))
        b = numeric(length(pred_full))
        samp = as.numeric(runif(NSAMP) > 0.5)
        a = ll_no_tri * samp + ll_full * (1 - samp)
        b = ll_no_tri * (1 - samp) + ll_full * samp
    
        diff = abs(sum(a) - sum(b))
        diffs[i] = diff
    }
    pr('\n\n')
    n_greater = sum(diffs > diff_tri)
    p_tri = (n_greater + 1) / (NSAMP + 1)
    
    pr(paste0('LL no unigram:            ', sum(ll_no_uni), '\n'), stdout())
    pr(paste0('LL no trigram:            ', sum(ll_no_tri), '\n'), stdout())
    pr(paste0('LL full:                  ', sum(ll_full), '\n'), stdout())
    pr('\n', stdout())
    pr(paste0('LL delta unigram:         ', diff_uni, '\n'), stdout())
    pr(paste0('p unigram:                ', p_uni, '\n'), stdout())
    pr('\n', stdout())
    pr(paste0('LL delta trigram:         ', diff_tri, '\n'), stdout())
    pr(paste0('p trigram:                ', p_tri, '\n'), stdout())
    pr('\n\n\n', stdout())

}



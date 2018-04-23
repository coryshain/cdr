library(lme4)
library(optimx)

z. <- function(x) scale(x)

bobyqa <- lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=50000))
nlminb <- lmerControl(optimizer="optimx",optCtrl=list(method=c("nlminb"),maxit=50000))
opt <- nlminb

d <- read.csv('dundee_nhst/DTSR_both/X_conv.csv', sep=' ', header=T)
form_baseline <- log(fdurGP) ~ z.(DiracDelta.s.sentpos.) + z.(ShiftedGammaKgt1.rate) + z.(ShiftedGammaKgt1.s.wdelta.) + z.(ShiftedGammaKgt1.s.wlen.) + z.(ShiftedGammaKgt1.s.prevwasfix.) + z.(ShiftedGammaKgt1.s.fwprob5surp.) + z.(ShiftedGammaKgt1.s.cumfwprob5surp.) + (1 + z.(DiracDelta.s.sentpos.) + z.(ShiftedGammaKgt1.rate) + z.(ShiftedGammaKgt1.s.wdelta.) + z.(ShiftedGammaKgt1.s.wlen.) + z.(ShiftedGammaKgt1.s.prevwasfix.) + z.(ShiftedGammaKgt1.s.fwprob5surp.) + z.(ShiftedGammaKgt1.s.cumfwprob5surp.) + z.(ShiftedGammaKgt1.s.gcg15surp.) + z.(ShiftedGammaKgt1.s.nodashtagssurp.) | subject)
m_baseline <- lmer(form_baseline, d, REML=F, control = opt)
print('Baseline')
print(summary(m_baseline))
save(m_baseline, file='lme_pcfg_baseline.rdata')

form_gcg <- log(fdurGP) ~ z.(DiracDelta.s.sentpos.) + z.(ShiftedGammaKgt1.rate) + z.(z.(ShiftedGammaKgt1.s.wdelta.)) + z.(ShiftedGammaKgt1.s.wlen.) + z.(ShiftedGammaKgt1.s.prevwasfix.) + z.(ShiftedGammaKgt1.s.fwprob5surp.) + z.(ShiftedGammaKgt1.s.cumfwprob5surp.) + z.(ShiftedGammaKgt1.s.gcg15surp.) + (1 + z.(DiracDelta.s.sentpos.) + z.(ShiftedGammaKgt1.rate) + z.(ShiftedGammaKgt1.s.wdelta.) + z.(ShiftedGammaKgt1.s.wlen.) + z.(ShiftedGammaKgt1.s.prevwasfix.) + z.(ShiftedGammaKgt1.s.fwprob5surp.) + z.(ShiftedGammaKgt1.s.cumfwprob5surp.) + z.(ShiftedGammaKgt1.s.gcg15surp.) + z.(ShiftedGammaKgt1.s.nodashtagssurp.) | subject)
m_gcg <- lmer(form_gcg, d, REML=F, control = opt)
print('GCG')
print(summary(m_gcg))
save(m_gcg, file='lme_pcfg_gcg.rdata')

form_wsj <- log(fdurGP) ~ z.(DiracDelta.s.sentpos.) + z.(ShiftedGammaKgt1.rate) + z.(ShiftedGammaKgt1.s.wdelta.) + z.(ShiftedGammaKgt1.s.wlen.) + z.(ShiftedGammaKgt1.s.prevwasfix.) + z.(ShiftedGammaKgt1.s.fwprob5surp.) + z.(ShiftedGammaKgt1.s.cumfwprob5surp.) + z.(ShiftedGammaKgt1.s.nodashtagssurp.) + (1 + z.(DiracDelta.s.sentpos.) + z.(ShiftedGammaKgt1.rate) + z.(ShiftedGammaKgt1.s.wdelta.) + z.(ShiftedGammaKgt1.s.wlen.) + z.(ShiftedGammaKgt1.s.prevwasfix.) + z.(ShiftedGammaKgt1.s.fwprob5surp.) + z.(ShiftedGammaKgt1.s.cumfwprob5surp.) + z.(ShiftedGammaKgt1.s.gcg15surp.) + z.(ShiftedGammaKgt1.s.nodashtagssurp.) | subject)
m_wsj <- lmer(form_wsj, d, REML=F, control = opt)
print('WSJ')
print(summary(m_wsj))
save(m_wsj, file='lme_pcfg_ptb.rdata')

form_both <- log(fdurGP) ~ z.(DiracDelta.s.sentpos.) + z.(ShiftedGammaKgt1.rate) + z.(ShiftedGammaKgt1.s.wdelta.) + z.(ShiftedGammaKgt1.s.wlen.) + z.(ShiftedGammaKgt1.s.prevwasfix.) + z.(ShiftedGammaKgt1.s.fwprob5surp.) + z.(ShiftedGammaKgt1.s.cumfwprob5surp.) + z.(ShiftedGammaKgt1.s.gcg15surp.) + z.(ShiftedGammaKgt1.s.nodashtagssurp.) + (1 + z.(DiracDelta.s.sentpos.) + z.(ShiftedGammaKgt1.rate) + z.(ShiftedGammaKgt1.s.wdelta.) + z.(ShiftedGammaKgt1.s.wlen.) + z.(ShiftedGammaKgt1.s.prevwasfix.) + z.(ShiftedGammaKgt1.s.fwprob5surp.) + z.(ShiftedGammaKgt1.s.cumfwprob5surp.) + z.(ShiftedGammaKgt1.s.gcg15surp.) + z.(ShiftedGammaKgt1.s.nodashtagssurp.) | subject)
m_both <- lmer(form_both, d, REML=F, control = opt)
print('Both')
print(summary(m_both))
save(m_both, file='lme_pcfg_both.rdata')

print('ANOVA:')
print(anova(m_baseline, m_gcg))
print(anova(m_baseline, m_wsj))
print(anova(m_gcg, m_both))
print(anova(m_wsj, m_both))




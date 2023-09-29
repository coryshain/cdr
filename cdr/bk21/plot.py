import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata, gaussian_kde, zscore, t as tdist
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

from cdr.plot import plot_irf
from cdr.bin.plot_surp_density import plot_density


def calc_kde(data):
    return kde(data.T)


def z(a, axis=-1):
    return (a - np.expand_dims(a.mean(axis=axis), axis=axis)) / a.std(axis=axis, keepdims=True)


def corr(a, b, axis=-1, spearman=True):
    a = np.asarray(a)
    b = np.asarray(b)
    if spearman:
        a = rankdata(a, axis=axis, method='min')
        b = rankdata(b, axis=axis, method='min')
    a = zscore(a, axis=axis) / np.sqrt(a.shape[axis])
    b = zscore(b, axis=axis) / np.sqrt(a.shape[axis])
    return (a * b).sum(axis=axis)

lms = [
    'cloze',
    'trigram',
    'gpt2',
    'gpt2region',
]
fns = [
    '%s',
    '%sprob'
]
dv_by_experiment = {'spr': 'SUM_3RT_trimmed', 'naming': 'TRIM_RT'}
scol_by_experiment = {'spr': 'SUB', 'naming': 'subject'}

if os.path.exists('bk21_results_path.txt'):
    with open('bk21_results_path.txt') as f:
        for line in f:
           line = line.strip()
           if line:
               results_dir = line
               break
else:
    results_dir = 'results/bk21'

# Predictability difference plots
spr = pd.read_csv('bk21_data/bk21_spr.csv')
naming = pd.read_csv('bk21_data/bk21_naming.csv')
spr_median = spr.groupby(['ITEM', 'condition'])[dv_by_experiment['spr']].median().reset_index()
df = spr.drop_duplicates(['ITEM', 'condition'])
df = pd.merge(df[[x for x in df.columns if not x == dv_by_experiment['spr']]], spr_median, on=['ITEM', 'condition'])
cloze = rankdata(df['clozeprob'], method='min')
gpt = rankdata(df['gpt2prob'], method='min')

## Line plot
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.xaxis.set_ticks_position('none')
x = [0, 1]
y = np.stack([cloze, gpt], axis=1)
gpt_v_cloze = gpt - cloze
df['gpt_v_cloze'] = gpt_v_cloze
vcenter = 0
vmin = gpt_v_cloze.min() - 1e-8
vmax = gpt_v_cloze.max() + 1e-8
if vmin < 0 and vmax > 0:
    bound = max(abs(vmin), vmax)
    vmin = -bound
    vmax = bound
elif vmin < 0.:
    vmax = 1e-8
else: # vmax > 0
    vmin = -1e-8
norm = matplotlib.colors.TwoSlopeNorm(
    vmin=vmin,
    vcenter=vcenter,
    vmax=vmax
)
cmap = plt.get_cmap('coolwarm')
color = norm(gpt_v_cloze)
color = cmap(color)
for _y, _c in zip(y, color):
    plt.plot(x, _y, lw=0.5, alpha=0.5, color=_c)
plt.xticks(ticks=[0, 1], labels=['Cloze', 'GPT-2'])
plt.ylabel('Item Rank')
plt.gcf().set_size_inches((2,4))
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'bk21/item_ranks_cloze_v_gpt.pdf'))
plt.close('all')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
x = gpt_v_cloze
y = df[dv_by_experiment['spr']]
a, b = np.polyfit(x, y, 1)
line = a * x + b
r = corr(x, y, spearman=False)
t = r * np.sqrt(len(x) - 2) / np.sqrt(1 - r ** 2)
p = (1 - tdist.cdf(np.abs(t), len(x) - 2)) * 2
stars = 0
if p < 0.001:
   stars = 3
elif p < 0.01:
   stars = 2
elif p < 0.05:
   stars = 1
plt.scatter(x, y, s=1, color='k')
plt.plot(x, line, color='k')
plt.text(x.min() + 50, y.max(), 'r = %.02f%s' % (r, '*' * stars))
plt.xlabel('Change in Rank (Cloze to GPT)')
plt.ylabel('Median RT (SPR)')
plt.gcf().set_size_inches((3,3))
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'item_ranks_rankchange_v_spr.pdf'))
plt.close('all')

## Scatterplot
nbin = 10
dv = dv_by_experiment['spr']
qmin, qmax = np.quantile(spr[dv], [0.1, 0.9])
cloze = spr['cloze']
cloze_bw = (cloze.max() - cloze.min()) / nbin
cloze_scale = 1. / cloze_bw
spr['xbin'] = ((cloze * cloze_scale).astype(int).clip(upper=nbin-1) + 0.5) / cloze_scale
gpt = spr['gpt2']
gpt_bw = (gpt.max() - gpt.min()) / nbin
gpt_scale = 1. / gpt_bw
spr['ybin'] = ((gpt * gpt_scale).astype(int).clip(upper=nbin-1) + 0.5) / gpt_scale
response = spr.groupby(['xbin', 'ybin'])[dv].median().reset_index()
response['counts'] = spr.groupby(['xbin', 'ybin'])[dv].count().reset_index(drop=True)
xbins = sorted(list(spr.xbin.unique()))
ybins = sorted(list(spr.ybin.unique()))
fig = plt.figure()
ax = fig.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
norm = matplotlib.colors.TwoSlopeNorm(
    vmin=response[dv].min() - 1,
    vcenter=response[dv].min(),
    vmax=response[dv].max()
)
color = norm(response[dv])
cmap = plt.get_cmap('coolwarm')
color = cmap(color)
ax.set_xlabel('Cloze Surprisal')
ax.set_ylabel('GPT-2 Surprisal')
ax.plot([xbins[0], xbins[-1]], [ybins[0], ybins[-1]], color='k', zorder=1)
ax.scatter(response.xbin, response.ybin, color=color, s=response.counts / 5, alpha=0.9, zorder=2)
fig.set_size_inches((4,4))
fig.tight_layout()
fig.savefig(os.path.join(results_dir, 'cloze_v_gpt_scatter.pdf'))
plt.close('all')

## Marginal data density by surprisal
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.scatter(spr['cloze'], spr[dv], alpha=0.01, s=0.5, color='k')
plt.xlabel('Cloze Surprisal')
plt.ylabel('SPR RT (z)')
plt.gcf().set_size_inches((5,2))
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'cloze_marginal.png'), dpi=1000)
plt.close('all')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
gpt_marg = spr.groupby('ybin')[dv].mean().reset_index()
plt.scatter(spr['gpt2'], spr[dv], alpha=0.01, s=0.5, color='k')
plt.xlabel('GPT-2 Surprisal')
plt.ylabel('SPR RT (z)')
plt.gcf().set_size_inches((5,2))
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'gpt_marginal.png'), dpi=1000)
plt.close('all')

## Text extraction
K = 10
print('Cloze vs. GPT-2 probability')
print('Spearman correlation: %0.2f' % np.corrcoef(df['clozeprob'], df['gpt2prob'])[0,1])
print('Mean rank change: %0.2f' % gpt_v_cloze.mean())
print('Top-%d positive rank changes from Cloze to GPT:' % K)
df = df.sort_values('gpt_v_cloze', ascending=False)
changes = df.gpt_v_cloze.iloc[:K].values.tolist()
words = df.critical_word.iloc[:K].values.tolist()
sents = df.words.iloc[:K].values.tolist()
df.iloc[:K].to_csv('bk21_data/cloze_v_gpt_predictability_difference_pos.csv', index=False)
for change, word, sent in zip(changes, words, sents):
    print('  Rank change: %d  |  %s  |  %s' % (change, word, sent))
print()
df = df.sort_values('gpt_v_cloze')
changes = df.gpt_v_cloze.iloc[:K].values.tolist()
words = df.critical_word.iloc[:K].values.tolist()
sents = df.words.iloc[:K].values.tolist()
df.iloc[:K].to_csv('bk21_data/cloze_v_gpt_predictability_difference_neg.csv', index=False)
print('Top-%d negative rank changes from Cloze to GPT:' % K)
for change, word, sent in zip(changes, words, sents):
    print('  Rank change: %d  |  %s  |  %s' % (change, word, sent))
print()

# Density plots
for experiment in ('spr', 'naming'):
    df = pd.read_csv('bk21_data/bk21_%s.csv' % experiment)
    for lm in lms:
        for fn in fns:
            name = fn % lm
            d = df[name]
            xmin, xmax = d.min(), d.max()
            out_name = '%s_%s_density.pdf' % (experiment, name)
            plot_density(d, results_dir, out_name, frac=(xmin, xmax))
            for condition in ('LC', 'MC', 'HC'):
                _df = df[df.condition == condition]
                d = _df[name]
                out_name = '%s_%s_%s_density.pdf' % (experiment, name, condition)
                plot_density(d, results_dir, out_name, frac=(xmin, xmax), size=(2, 0.25))
    name = dv_by_experiment[experiment]
    for support in (0.8, 1):
        d = df[name]
        if support == 1:
            xmin, xmax = d.min(), d.max()
            support_name = ''
        else:
            xmin, xmax = np.nanquantile(d, (0.1, 0.9))
            support_name = '_0.8'
        out_name = '%s_%s%s_density.pdf' % (experiment, name, support_name)
        color = (0., 0., 1., 0.5)
        plot_density(d, results_dir, out_name, frac=(xmin, xmax), color=color)
        for condition in ('LC', 'MC', 'HC'):
            _df = df[df.condition == condition]
            d = _df[name]
            out_name = '%s_%s_%s%s_density.pdf' % (experiment, name, condition, support_name)
            plot_density(d, results_dir, out_name, frac=(xmin, xmax), color=color, size=(2, 0.25))

# Curvature plots
for path in os.listdir(results_dir):
    if path.endswith('_plot.csv'):
        csv_path = os.path.join(results_dir, path)
        if os.path.exists(csv_path):
            plot_data = pd.read_csv(csv_path)
            for lm in lms:
                for fn in fns:
                    name = fn % lm
                    if '%s_x' % name in plot_data:
                        plot_x = plot_data['%s_x' % name].values[..., None]
                        plot_y = plot_data['%s_y' % name].values[..., None]
                        err = plot_data['%s_err' % name].values[..., None]
                        lq = plot_y - err
                        uq = plot_y + err

                        plot_irf(
                            plot_x,
                            plot_y,
                            [name],
                            lq=lq,
                            uq=uq,
                            outdir=results_dir,
                            filename=path[:-4] + '_%s.pdf' % name,
                            legend=False,
                            plot_x_inches=2,
                            plot_y_inches=1.5,
                            x_max_ticks=2,
                            dpi=150
                        )
        


import os
import pandas as pd
from cdr.plot import plot_irf

lms = [
    'ngram',
    'totsurp',
    'gpt',
    'gptj',
    'gpt3',
    'cloze'
]

if os.path.exists('gam_results_path.txt'):
    with open('gam_results_path.txt') as f:
        for line in f:
           line = line.strip()
           if line:
               results_dir = line
               break
else:
    results_dir = 'results/gam'
for dataset in os.listdir(results_dir):
    for path in os.listdir(os.path.join(results_dir, dataset)):
        if path.endswith('_plot.csv'):
            for lm in lms:
                plot_data = pd.read_csv(os.path.join(results_dir, dataset, path))
                if lm == 'cloze':
                    lm = 'clozesurp'
                if '%s_x' % lm in plot_data:
                    plot_x = plot_data['%s_x' % lm].values[..., None]
                    plot_y = plot_data['%s_y' % lm].values[..., None]
                    err = plot_data['%s_err' % lm].values[..., None]
                    lq = plot_y - err
                    uq = plot_y + err

                    plot_irf(
                        plot_x,
                        plot_y,
                        [lm],
                        lq=lq,
                        uq=uq,
                        outdir=os.path.join(results_dir, dataset),
                        filename=path[:-4] + '.pdf',
                        legend=False,
                        plot_x_inches=2,
                        plot_y_inches=1.5,
                        dpi=150
                    )
        

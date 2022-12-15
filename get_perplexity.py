import sys
import numpy as np
import pandas as pd


def rename(x):
    if x == 'gpt2':
        return 'gpt'
    if x == 'gpt-j':
        return 'gptj'
    return x


datasets = [
    'brown',
    'dundee',
    'geco',
    'naturalstories',
    'provo',
]

LMs = [
    'ngram',
    'pcfg',
    'gpt',
    'gptj',
    'gpt3',
    'cloze'
]

out = []
data_path = '../data/TEXT_readingstimuli/%s'

for dataset in datasets:
    mb = pd.read_csv(data_path % ('%s.wsj02to21-gcg15-nol-prtrm-3sm-synproc-+c_+u_+b5000_parsed.all-itemmeasures' % dataset), sep=' ')
    meister = pd.read_csv(data_path % ('cory_%s.tsv' % dataset), sep='\t').rename(rename, axis=1)
    gpt3 = pd.read_csv(data_path % ('%s.gpt3davinci.itemmeasures' % dataset), sep=' ')
    _out = []
    for LM in LMs:
        if LM == 'pcfg':
            val = -np.log(2 ** (-mb['totsurp'])).mean()
        elif LM == 'cloze':
            if 'clozesurp' in mb:
                val = mb['clozesurp'].mean()
            else:
                val = np.nan
        elif LM == 'gpt3':
            val = gpt3['gpt3surp'].mean()
        else:
            val = meister[LM].mean()
        val = np.exp(val)
        _out.append((LM, val))
    _out = pd.DataFrame(_out, columns=['LM', 'perplexity'])
    _out['dataset'] = dataset
    out.append(_out)
out = pd.concat(out, axis=0)
out.to_csv(sys.stdout, index=False)


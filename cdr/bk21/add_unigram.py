import sys
import os
import numpy as np
import pandas as pd

from cdr.util import stderr

# Get user-defined path to B&K21 data
base_path = 'data'
model_path = os.path.join(base_path, 'MODEL_kenlm', 'openwebtext.1.fw.kenlm')
items_path = os.path.join('bk21_data', 'items.csv')
if not os.path.exists(items_path):
    stderr('Items file for Brothers & Kuperberg 2021 no found. Run `python -m cdr.bk21.get_items` first.\n')
    exit()

items = pd.read_csv(items_path)
pos = items.critical_word_pos.values - 1
pos_cols = sorted([col for col in items if col.startswith('word') and col != 'words'])
words = items[pos_cols].values
c1 = words[np.arange(len(pos)), pos]
c2 = words[np.arange(len(pos)), pos + 1]
c3 = words[np.arange(len(pos)), pos + 2]
lines = np.stack([c1, c2, c3], axis=1).tolist()

model = {}
with open(model_path, 'r') as f:
    for line in f:
        if len(line) and line[0] == '-':
            sline = line.strip().split()
            model[sline[1]] = float(sline[0])

unigram1 = []
length1 = []
unigram3 = []
length3 = []
for line in lines:
    val = None
    length = None
    for word in line:
        if word in model:
            cur = -model[word]
        else:
            cur = -model['<unk>']
        if val is None:  # First (i.e., critical) word
            val = cur
            unigram1.append(val)
            length = len(word)
            length1.append(length)
        else:
            val += cur
            length += len(word)
    unigram3.append(val)
    length3.append(length)

items['unigram'] = unigram1
items['unigramregion'] = unigram3
items['wlen'] = length1
items['wlenregion'] = length3

items.to_csv(items_path, index=False)


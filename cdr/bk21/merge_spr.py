import os
import numpy as np
import pandas as pd

key_cols_items = [
   'group',
   'condition'
]

key_cols_spr = [
    'ITEM',
    'condition'
]

NFOLDS = 5

# Set path to B&K21 data
base_path = 'data/SPR_brotherskuperberg/orig'
spr_path = os.path.join(base_path, 'SPRT_LogLin_216.csv')

# Load items
items_path = os.path.join('bk21_data', 'items.csv')
if not os.path.exists(items_path):
    stderr('Items file for Brothers & Kuperberg 2021 no found. Run `python -m cdr.bk21.get_items` first.\n')
    exit()
out_path = os.path.join('bk21_data', 'bk21_spr.csv')
items = pd.read_csv(items_path)

# Load SPR data
spr = pd.read_csv(spr_path)

# Compute 2-fold CV folds
items['fold'] = (items['itemnum'] % NFOLDS) + 1

# Get merge columns
item_cols = [col for col in items if not (col in spr or col in key_cols_items)] + key_cols_items

# Process cloze
spr['clozeprob'] = spr.cloze.where(spr.cloze > 0, 0.5 / 90)  # B&K gave half a count (out of an average 90 completions per item) for items with cloze 0
spr['cloze'] = -np.log(spr.clozeprob)  

# Process cloze
spr['trigramprob'] = spr.trigram
spr['trigram'] = -np.log(spr.trigramprob)  

# Merge and save
out = pd.merge(spr, items[item_cols], left_on=key_cols_spr, right_on=key_cols_items)
out = out.sort_values(['SUB', 'ITEM', 'condition'], key=lambda x: x.map({'LC': 0, 'MC': 2, 'HC': 3}) if x.name == 'condition' else x)
assert len(out) == len(spr), 'Dataset changed size during merge. Input had %d rows, output has %d' % (len(spr), len(out))
out['sortix'] = np.arange(len(out))
out.to_csv(out_path, index=False)


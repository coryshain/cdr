import os
import numpy as np
import pandas as pd

key_cols_items = [
   'naming_group_num',
   'condition'
]

key_cols_naming = [
    'item',
    'condition'
]

NFOLDS = 5

# Set path to B&K21 data
base_path = 'data/SPR_brotherskuperberg/orig'
naming_path = os.path.join(base_path, 'LogLin_Naming.csv')

# Load items
items_path = os.path.join('bk21_data', 'items.csv')
if not os.path.exists(items_path):
    stderr('Items file for Brothers & Kuperberg 2021 no found. Run `python -m cdr.bk21.get_items` first.\n')
    exit()
out_path = os.path.join('bk21_data', 'bk21_naming.csv')

items = pd.read_csv(items_path)

# Load naming data
naming = pd.read_csv(naming_path)
naming = naming.rename(lambda x: 'log_trigram' if x == 'log_traigram' else x)

# Compute 2-fold CV folds
items['fold'] = (items['itemnum'] % NFOLDS) + 1

# Get merge columns
item_cols = [col for col in items if not (col in naming or col in key_cols_items)] + key_cols_items

# Process cloze
naming['clozeprob'] = naming.cloze.where(naming.cloze > 0, 0.5 / 90)  # B&K gave half a count (out of an average 90 completions per item) for items with cloze 0
naming['cloze'] = -np.log(naming.clozeprob)

# Process cloze
naming['trigramprob'] = naming.trigram
naming['trigram'] = -np.log(naming.trigramprob)

# Naming dataset uses a different group numbering convention than SPR dataset, so infer naming item numbers for merge
target_nums = naming[['item', 'critical_word']].drop_duplicates().sort_values('critical_word').reset_index(drop=True)
critical_words = items[items.used_in_expt2 > 0][['critical_word']].drop_duplicates().reset_index(drop=True)
assert (target_nums.critical_word == critical_words.critical_word).sum() == 84, 'Error inferring naming expt item numbers'
critical_words['naming_group_num'] = target_nums.item
n_items = len(items)
items = pd.merge(items, critical_words, how='left', on='critical_word')
assert len(items) == n_items, 'Error inferring naming expt item numbers. Merged changed items from %d to %d.' % (n_items, len(items))
test = pd.merge(naming, items, left_on=key_cols_naming, right_on=key_cols_items)

# Merge and save
out = pd.merge(naming, items[item_cols], left_on=key_cols_naming, right_on=key_cols_items)
out = out.sort_values(['subject', 'item', 'condition'], key=lambda x: x.map({'LC': 0, 'MC': 2, 'HC': 3}) if x.name == 'condition' else x)
assert len(out) == len(naming), 'Dataset changed size during merge. Input had %d rows, output has %d' % (len(naming), len(out))
out['sortix'] = np.arange(len(out))
out.to_csv(out_path, index=False)



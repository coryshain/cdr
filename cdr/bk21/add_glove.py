import sys
import os
import io
import zipfile
import re
import numpy as np
from numpy.linalg import norm
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import nltk
from nltk.corpus import stopwords

from cdr.util import stderr

def corr(a, b, rtype='spearman'):
    if rtype == 'spearman':
        return spearmanr(a, b)[0]
    if rtype == 'pearson':
        return pearsonr(a, b)[0,1]
    raise ValueError('Unrecognized correlation type %s' % rtype)

nltk.download('stopwords')
stopwords = stopwords.words('english')
punctuation = '''[.,:;!?'"`’]'''

glove_zip_path = 'data/TEXT_glove/glove.840B.300d.zip'

items_path = os.path.join('bk21_data', 'items.csv')
if not os.path.exists(items_path):
    stderr('Items file for Brothers & Kuperberg 2021 no found. Run `python -m cdr.bk21.get_items` first.\n')

items = pd.read_csv(items_path)
words = items['words'].str.split().values.tolist()
critical_word_positions = (items['critical_word_pos'].values - 1).tolist()  # Subtract 1 because 1-indexed
for i, (sent, pos) in enumerate(zip(words, critical_word_positions)):
    cleaned = []
    for j, word in enumerate(sent):
        word = re.sub(punctuation, '', word)
        if word.lower() not in stopwords and j < pos:
            cleaned.append(word)
    words[i] = cleaned
keys = items['itemnum'].values.tolist()
sents = {k: s for k, s in zip(keys, words)}
targets = {k: t for k, t in zip(keys, items['critical_word'].values.tolist())}

glove_table = {}
with zipfile.ZipFile(glove_zip_path, 'r') as zf:
    with io.TextIOWrapper(zf.open('glove.840B.300d.txt', 'r'), encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 10000 == 0:
                stderr('\rProcessed %d vocab items...' % i)
            line = line.strip().split(' ')
            if line:
                glove_table[line[0]] = np.array([float(x) for x in line[1:]])
stderr('\n')

glove_dist = []
for k in sents:
    sent = sents[k]
    target = targets[k]
    target_glove = glove_table[target]
    sim = []
    for word in sent:  # Sents are already filtered above to contain only preceding context
        context_glove = glove_table[word]
        sim.append(corr(target_glove, context_glove, rtype='spearman'))
    min_dist = 1 - np.array(sim).max()
    mean_dist = 1 - np.array(sim).mean()
    row = dict(itemnum=k, glovedistmin=min_dist, glovedistmean=mean_dist)
    glove_dist.append(row)
glove_dist = pd.DataFrame(glove_dist)

# Remove any previously computed GloVe features
for col in items:
     if 'glovedist' in col:
          del items[col]

items = pd.merge(items, glove_dist, on=['itemnum'])
items.to_csv(items_path, index=False)




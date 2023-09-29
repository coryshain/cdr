import sys
import os
import pandas as pd
import pypandoc
from pypandoc.pandoc_download import download_pandoc

# download_pandoc()

# Set path to B&K21 data
base_path = 'data/SPR_brotherskuperberg/orig'
# Path to Word file containing item text
src_path = os.path.join(base_path, 'Stimuli_Appendix_format.docx')
# Path to CSV file containing summed RTS by item
spr_path = os.path.join(base_path, 'SPRT_LogLin_216.csv')
# Output directory for generated files
gen_dir_path = os.path.join('bk21_data')
# Output path for plain text extracted from Word file
txt_path = os.path.join(gen_dir_path, 'appendix_src.txt')
# Output path for by-item textual features
items_path = os.path.join(gen_dir_path, 'items.csv')
# Create output directory
if not os.path.exists(gen_dir_path):
    os.makedirs(gen_dir_path)

spr = pd.read_csv(spr_path)
items = spr.drop_duplicates(['ITEM', 'condition'])

output = pypandoc.convert_file(src_path, 'plain', outputfile=txt_path, extra_args=('--standalone','--wrap=none'))
assert output == '', '%s' % output

csv = []
with open(txt_path, 'r') as f:
    started = False
    i = 0
    for line in f:
        line = line.strip()
        if line.startswith('-'):
            started = True
            continue
        if line and started:
            expt2 = False
            if line.startswith('*'):
                line = line[1:]
                expt2 = True
            toks = line.split()
            # One token as a stray system character, fix
            toks = [x if not x.startswith('"boo!"') else '"boo!"' for x in toks]
            line = ' '.join(toks[1:-1])
            condition = toks[0][0] + 'C'
            clozeprob = int(toks[-1][1:-2]) / 100
            toks = toks[1:-1]
            row = dict(
                words=line,
                itemnum=i,
                used_in_expt2=expt2,
                condition=condition,
                clozeprob=clozeprob
                
            )
            for j in range(len(toks)):
                row['word%02d' % (j + 1)] = toks[j]
            csv.append(row)
            i += 1 
csv = pd.DataFrame(csv)

# No item numbers are given in the source materials, so they must be inferred from
# determining when a given critical word occurs in a given position. Unfortunately,
# in some cases, an item will match a word+position code even if that is not the
# critical word+position for that item. The following block addresses this issue
# by finding items that are flagged with multiple IDs and removing the incorrect one,
# which can be done by finding IDs that are assigned to more than 3 items and
# and removing those that also have other IDs.
group_ids = set()
critical_words = []
for critical_word, position, condition, item in items[['critical_word', 'position', 'condition', 'ITEM']].values:
    sel = (csv.condition == condition ) & (csv['word%02d' % position] == critical_word)
    group_ids.add(item)
    if 'IS_GROUP_%s' % item in csv:
        csv['IS_GROUP_%s' % item] += sel.astype(int)
    else:
        csv['IS_GROUP_%s' % item] = sel.astype(int)
    row = dict(
        critical_word=critical_word,
        critical_word_pos=position,
        group=item,
        condition=condition
    )
    critical_words.append(row)
critical_words = pd.DataFrame(critical_words)
csv['group'] = 0
for item in group_ids:
    sel =csv['IS_GROUP_%s' % item]
    n = sel.sum()
    if n > 3:
        err = sel & (csv[[x for x in csv.columns if x.startswith('IS_GROUP')]].sum(axis=1) > 1)
        sel = sel & (~err)
        csv['IS_GROUP_%s' % item] = sel
    csv['group'] += item * sel
for item in group_ids:
    del csv['IS_GROUP_%s' % item]

# Merge in item number from the source now that group has been computed
csv = pd.merge(csv, critical_words, on=['group', 'condition'])
csv = csv.sort_values('itemnum')

# Save item features
csv.to_csv(items_path, index=False)




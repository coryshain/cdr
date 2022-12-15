import sys

base = """color_by_sign: True
bold_signif: True
fdr_by_group: True
partitions:
  - test

response_order:
  - fdur
  - rt
  - fdurSPsummed
  - fdurFP
  - fdurGP
  - pooled

comparison_name_map:
  CDR_: ""
  cloze(_|$): Cloze$_{f(\\\\textsc{surp})}$
  ngram(_|$): \\\\textit{n}-gram$_{f(\\\\textsc{surp})}$
  pcfg(_|$): PCFG$_{f(\\\\textsc{surp})}$
  gpt(_|$): GPT-2$_{f(\\\\textsc{surp})}$
  gptj(_|$): GPT-J$_{f(\\\\textsc{surp})}$
  gpt3(_|$): GPT-3$_{f(\\\\textsc{surp})}$
  allLM(_|$): All-LMs$_{f(\\\\textsc{surp})}$
  cloze: Cloze
  ngram: \\\\textit{n}-gram
  pcfg: PCFG
  gpt: GPT-2
  gptj: GPT-J
  gpt3: GPT-3
  allLM: All-LMs
  v_: " vs.\\\\\\\\ "
  _v_: " vs.\\\\\\\\ "
  nosurp: $\\\\emptyset$
  prob: $_{\\\\textsc{prob}}$
  "0.50": $_{\\\\textsc{surp}^{1/2}}$
  "0.75": $_{\\\\textsc{surp}^{3/4}}$
  "1.00": $_{\\\\textsc{surp}^{1}}$
  "1.33": $_{\\\\textsc{surp}^{4/3}}$
  "2.00": $_{\\\\textsc{surp}^{2}}$
  gpt_v_gptpcfg: GPT-2\\+PCFG$_{f(\\\\textsc{surp})}$ vs.\\ GPT-2$_{f(\\\\textsc{surp})}$
  lin: $_{\\\\textsc{surp}^{1}}$
  sublin: $_{\\\\textsc{surp}^{<1}}$
  suplin: $_{\\\\textsc{surp}^{>1}}$
  notsublin: $_{\\\\textsc{surp}^{\\\\geq 1}}$
  notsuplin: $_{\\\\textsc{surp}^{\\\\leq 1}}$

group_order:
  - Cloze
  - \\textit{n}-gram
  - PCFG
  - GPT-2
  - GPT-J
  - GPT-3
  - All LMs
  - Between LMs

positive_only:
%s

groups:
%s
"""

out = base

models = [
    'cloze',
    'ngram',
    'pcfg',
    'gpt',
    'gptj',
    'gpt3',
]

groups = [
  'Cloze',
  '\\textit{n}-gram',
  'PCFG',
  'GPT-2',
  'GPT-J',
  'GPT-3',
  'All LMs',
  'Between LMs',
]

group2name = {
  'cloze': 'Cloze',
  'ngram': '\\textit{n}-gram',
  'pcfg': 'PCFG',
  'gpt': 'GPT-2',
  'gptj': 'GPT-J',
  'gpt3': 'GPT-3',
  'allLM': 'All LMs',
}

functions = [
    'prob',
    '0.50',
    '0.75',
    '1.00',
    '1.33',
    '2.00',
]

composite = [
    ('lin', 'sublin'),
    ('lin', 'suplin'),
    ('notsublin', 'sublin'),
    ('notsuplin', 'suplin'),
]

positive_only = ''
groups = ''

for model in models + ['allLM']:
    groups += '  %s:\n' % group2name[model]
    s = '%s_v_nosurp' % model
    groups += '    - %s\n' % s
    positive_only += '  - %s\n' % s
    for fn in functions:
        s = '%s_v_%s%s' % (model, model, fn)
        groups += '    - %s\n' % s
        positive_only += '  - %s\n' % s
    for fn in functions:
        s = '%s%s_v_nosurp' % (model, fn)
        groups += '    - %s\n' % s
        positive_only += '  - %s\n' % s
    for i, fn1 in enumerate(functions):
        if i < len(functions) - 1:
            for fn2 in functions[i+1:]:
                s = '%s%s_v_%s%s' % (model, fn2, model, fn1)
                groups += '    - %s\n' % s
    for composite_comparison in composite:
        a, b = composite_comparison
        s = '%s%s_v_%s%s' % (model, b, model, a)
        groups += '    - %s\n' % s
    if model == 'gpt':
        s = '%s_v_%snormal' % (model, model)
        groups += '    - %s\n' % s

groups += '  Between LMs:\n'
for i, m1 in enumerate(models):
    if i < len(models) - 1:
        for m2 in models[i+1:]:
            s = '%s_v_%s' % (m2, m1)
            groups += '    - %s\n' % s
s = 'gptpcfg_v_gpt'
groups += '    - %s\n' % s
positive_only += '  - %s\n' % s


sys.stdout.write(out % (positive_only, groups))
sys.stdout.flush()


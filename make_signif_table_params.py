import sys

base = """color_by_sign: True
bold_signif: True
partitions:
  - test

response_order:
  - fdur
  - rt
  - fdurSPsummed
  - fdurFP
  - fdurGP

comparison_name_map:
  main: Main

group_order:
  - Overall
  - Distributional

positive_only:
  - all

groups:
%s
"""

comparisons_main = [
    'main_v_main-gpt',
    'main_v_main-unigramsurpOWT',
    'main-unigramsurpOWT_v_main-unigramsurpOWT-gpt',
    'main-gpt_v_main-unigramsurpOWT-gpt',
]

comparisons_dist = [
    'yesinteraction_v_nointeraction',
    'dist_v_dist-unigramsurpOWTmu',
    'dist_v_dist-unigramsurpOWTsigma',
    'dist_v_dist-unigramsurpOWTbeta',
    'dist_v_dist-gptmu',
    'dist_v_dist-gptsigma',
    'dist_v_dist-gptbeta',
]

out = base

groups = '  Overall:\n'
for comparison in comparisons_main:
    groups += '    - %s\n' % comparison

groups += '  Distributional:\n'
for comparison in comparisons_dist: 
    groups += '    - %s\n' % comparison
    
sys.stdout.write(out % (groups))
sys.stdout.flush()


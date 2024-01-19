import os
import argparse

def pretty_table(row_collection, key_list, field_sep=' '):
  return '\n'.join([field_sep.join([str(row[col]).ljust(width)
    for (col, width) in zip(key_list, [max(map(len, column_vector))
      for column_vector in [ [v[k]
        for v in row_collection if k in v]
          for k in key_list ]])])
            for row in row_collection])

models = [
    'nosurp',
    'ngram',
    'ngramprob',
    'pow0_5_ngram_',
    'pow0_75_ngram_',
    'pow1_ngram_',
    'pow1_33_ngram_',
    'pow2_ngram_',
    'totsurp',
    'totsurpprob',
    'pow0_5_totsurp_',
    'pow0_75_totsurp_',
    'pow1_totsurp_',
    'pow1_33_totsurp_',
    'pow2_totsurp_',
    'gpt',
    'gptprob',
    'pow0_5_gpt_',
    'pow0_75_gpt_',
    'pow1_gpt_',
    'pow1_33_gpt_',
    'pow2_gpt_',
    'gptj',
    'gptjprob',
    'pow0_5_gptj_',
    'pow0_75_gptj_',
    'pow1_gptj_',
    'pow1_33_gptj_',
    'pow2_gptj_',
    'gpt3',
    'gpt3prob',
    'pow0_5_gpt3_',
    'pow0_75_gpt3_',
    'pow1_gpt3_',
    'pow1_33_gpt3_',
    'pow2_gpt3_',
]

model_to_name = {
    'nosurp': '$\empty_set$',
    'ngram': '$n$-gram$_{f(\\textsc{surp})}$',
    'ngramprob': '$n$-gram${_\\textsc{prob}}$',
    'pow0_5_ngram_': '$n$-gram$_{\\textsc{surp}^{1/2}}$',
    'pow0_75_ngram_': '$n$-gram$_{\\textsc{surp}^{3/4}}$',
    'pow1_ngram_': '$n$-gram$_{\\textsc{surp}^{1}}$',
    'pow1_33_ngram_': '$n$-gram$_{\\textsc{surp}^{4/3}}$',
    'pow2_ngram_': '$n$-gram$_{\\textsc{surp}^{2}}$',
    'totsurp': 'PCFG$_{f(\\textsc{surp})}$',
    'totsurpprob': 'PCFG${_\\textsc{prob}}$',
    'pow0_5_totsurp_': 'PCFG$_{\\textsc{surp}^{1/2}}$',
    'pow0_75_totsurp_': 'PCFG$_{\\textsc{surp}^{3/4}}$',
    'pow1_totsurp_': 'PCFG$_{\\textsc{surp}^{1}}$',
    'pow1_33_totsurp_': 'PCFG$_{\\textsc{surp}^{4/3}}$',
    'pow2_totsurp_': 'PCFG$_{\\textsc{surp}^{2}}$',
    'gpt': 'GPT-2$_{f(\\textsc{surp})}$',
    'gptprob': 'GPT-2${_\\textsc{prob}}$',
    'pow0_5_gpt_': 'GPT-2$_{\\textsc{surp}^{1/2}}$',
    'pow0_75_gpt_': 'GPT-2$_{\\textsc{surp}^{4/3}}$',
    'pow1_gpt_': 'GPT-2$_{\\textsc{surp}^{1}}$',
    'pow1_33_gpt_': 'GPT-2$_{\\textsc{surp}^{4/3}}$',
    'pow2_gpt_': 'GPT-2$_{\\textsc{surp}^{2}}$',
    'gptj': 'GPT-J$_{f(\\textsc{surp})}$',
    'gptjprob': 'GPT-J${_\\textsc{prob}}$',
    'pow0_5_gptj_': 'GPT-J$_{\\textsc{surp}^{1/2}}$',
    'pow0_75_gptj_': 'GPT-J$_{\\textsc{surp}^{3/4}}$',
    'pow1_gptj_': 'GPT-J$_{\\textsc{surp}^{1}}$',
    'pow1_33_gptj_': 'GPT-J$_{\\textsc{surp}^{4/3}}$',
    'pow2_gptj_': 'GPT-J$_{\\textsc{surp}^{2}}$',
    'gpt3': 'GPT-3$_{f(\\textsc{surp})}$',
    'gpt3prob': 'GPT-3${_\\textsc{prob}}$',
    'pow0_5_gpt3_': 'GPT-3$_{\\textsc{surp}^{1/2}}$',
    'pow0_75_gpt3_': 'GPT-3$_{\\textsc{surp}^{3/4}}$',
    'pow1_gpt3_': 'GPT-3$_{\\textsc{surp}^{1}}$',
    'pow1_33_gpt3_': 'GPT-3$_{\\textsc{surp}^{4/3}}$',
    'pow2_gpt3_': 'GPT-3$_{\\textsc{surp}^{2}}$',
}

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''Show table of GAM model performance''')
    argparser.add_argument('corpus', help='Name of corpus to show (one of "brown", "dundee", "geco", "natstor", "natstormaze", "provo")')
    args = argparser.parse_args()
    
    corpus = args.corpus
    if os.path.exists('gam_results_path.txt'):
        with open('gam_results_path.txt') as f:
            for line in f:
               line = line.strip()
               if line:
                   results_path = line
                   break
    else:
        results_path = 'results/gam'
    base_path = os.path.join(results_path, '{corpus}'.format(corpus=corpus)
    results = {}
    
    for path in os.listdir(base_path):
        if path.endswith('.txt'):
            with open(os.path.join(base_path, path), 'r') as f:
                rv = None
                model = None
                ll = None
                for line in f:
                    line = line.strip()
                    if line.startswith('Model path:'):
                        name_pieces = line.split()[-1].split('/')[-1][:-6].split('_')
                        rv = name_pieces[0]
                        model ='_'.join(name_pieces[1:])
                    elif line.startswith('Loglik:'):
                        ll = int(round(float(line.split()[-1])))
                assert rv is not None, 'No response variable found'
                assert model is not None, 'No model found'
                assert ll is not None, 'No likelihood value found'
                
                if rv not in results:
                    results[rv] = {}
                assert model not in results[rv], 'Duplicate value found for response %s, model %s' % (rv, model)
                results[rv][model] = ll
    
    for rv in results:
        table = [dict(model='model', loglik='loglik')] + [dict(model=model, loglik=str(results[rv][model])) for model in models]
        print(rv)
        print(pretty_table(table, ['model', 'loglik']))
        print()
    
    

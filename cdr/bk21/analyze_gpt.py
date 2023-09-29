import sys
import os
import numpy as np
import pandas as pd
import transformers
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

ADD_BOS = True
K = 10

model_id = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_id)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

items_path = os.path.join('bk21_data', 'cloze_v_gpt_predictability_difference_neg.csv')
if not os.path.exists(items_path):
    stderr('Analysis data file no found. Run `python -m cdr.bk21.plot` first.\n')
    exit()

items = pd.read_csv(items_path)
words = items['words'].str.split().values.tolist()
critical_word_positions = (items['critical_word_pos'].values - 1).tolist()  # Subtract 1 because 1-indexed
critical_words = items['critical_word'].tolist()
keys = items['itemnum'].values.tolist()
prefixes = []
targets1 = []
targets3 = []
for key, sent, pos, critical_word in zip(keys, words, critical_word_positions, critical_words):
    prefixes.append(' '.join(sent[:pos]))
    target_toks = sent[pos:pos+3]
    assert target_toks[0] == critical_word, 'Error on item %d: expected critical word "%s", found "%s".' % (key, critical_word, target_toks[0])
    targets1.append(' ' + target_toks[0])         # Single critical target word, space needs to be added beforehand for the tokenizer
    targets3.append(' ' + ' '.join(target_toks))  # 3-word critical region, space needs to be added beforehand for the tokenizer

_prefixes = prefixes

surp = []
with torch.no_grad():
    # Tokenize and encode
    prefixes = tokenizer(prefixes)['input_ids']
    targets1 = tokenizer(targets1)['input_ids']
    targets3 = tokenizer(targets3)['input_ids']

    # Construct prompts and labels
    prompts = []
    labels1 = []
    labels3 = []
    attention_mask = []
    max_len = 0
    for prefix, target1, target3 in zip(prefixes, targets1, targets3):
        if ADD_BOS:
            prefix = [tokenizer.bos_token_id] + prefix
        npad = len(prefix) - 1  # Don't predict the first token
        ndiff = len(target3) - len(target1)
        prompt = prefix + target3[:-1]
        prompts.append(prompt)
        labels1.append([-100] * npad + target1 + [-100] * ndiff)
        labels3.append([-100] * npad + target3)
        attention_mask.append([1] * len(prompt))
        max_len = max(max_len, len(prompt))

    # Pad
    for i in range(len(prompts)):
        prompt = prompts[i]
        npad = max_len - len(prompt)
        prompts[i] = prompt + [0] * npad
        labels1[i] = labels1[i] + [-100] * npad
        labels3[i] = labels3[i] + [-100] * npad
        attention_mask[i] = attention_mask[i] + [0] * npad

    # Send to device
    prompts = torch.tensor(prompts).to(model.device)
    labels1 = torch.tensor(labels1).to(model.device)
    labels3 = torch.tensor(labels3).to(model.device)
    attention_mask = torch.tensor(attention_mask).float().to(model.device)

    # Run model
    surprisals1 = []
    surprisals3 = []
    topk = {}

    # Get batch
    _prompts = prompts.contiguous()
    _labels1 = labels1.contiguous()
    _labels3 = labels3.contiguous()
    _attention_mask = attention_mask
    
    # Call on batch
    outputs = model(_prompts, attention_mask=_attention_mask)
    logits = outputs['logits'].contiguous()
    logits[:, :, tokenizer.pad_token_id] = -float("inf")
    preds = logits.argmax(axis=-1) * _attention_mask.int() # * (_labels1 >= 0).int()
    logits = logits.permute((0, 2, 1))

    ranks = torch.argsort(logits, axis=1, descending=True)
    pos_mask_ix = torch.argmax((labels1 != -100).int(), axis=1)
    pos_mask = torch.zeros_like(labels1).scatter_(1, pos_mask_ix.unsqueeze(1), 1.)
    # pos_mask = (_labels1 != -100).int()
    predsk = []
    probsk = []
    wordsk= []
    for k in range(K):
        predk = ranks[:, k, :] * pos_mask
        predk = predk * _attention_mask.int()
        probk = np.asarray(torch.exp(-(torch.nn.CrossEntropyLoss(reduction='none')(logits, predk) * pos_mask).sum(axis=1)).contiguous().cpu())
        wordk = []
        for i, _predk in enumerate(predk):
            wordk.append(tokenizer.decode(_predk).replace('!', ''))

        predsk.append(predk)
        probsk.append(probk)
        wordsk.append(wordk)

    wordsk_by_sent = [[] for _ in range(len(items))]
    probsk_by_sent = [[] for _ in range(len(items))]
    for k, (_wordsk, _probsk) in enumerate(zip(wordsk, probsk)):
        for s, (_wordk, _probk) in enumerate(zip(_wordsk, _probsk)):
            wordsk_by_sent[s].append("``%s''" % _wordk)
            probsk_by_sent[s].append(_probk)

    for i, (_prefix, _cloze, _surp, word, _predk, _probsk) in enumerate(zip(_prefixes, items.clozeprob.values, items.gpt2prob.values, items.critical_word.values, wordsk_by_sent, probsk_by_sent)):
        print('  \\hline')
        print('  Critical word: & \multicolumn{2}{l}{%s}\\\\' % word)
        print('  Cloze probability: & %0.5f\\\\' % _cloze)
        print('  GPT-2 probability: & %0.5f\\\\' % _surp)
        print('  Prefix: & \multicolumn{2}{p{0.9\\textwidth}}{%s}\\\\' % _prefix)
        print('  Top-%d GPT-2 predictions: & Token & $p$\\\\' % K)
        for __predk, _probk in zip(_predk, _probsk):
            print('  & %s & %0.4f\\\\' % (__predk, _probk))
        print()


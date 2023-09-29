import sys
import os
import numpy as np
import pandas as pd
import transformers
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

BATCH_SIZE = 4
ADD_BOS = True

model_id = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_id)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

items_path = os.path.join('bk21_data', 'items.csv')
if not os.path.exists(items_path):
    stderr('Items file for Brothers & Kuperberg 2021 no found. Run `python -m cdr.bk21.get_items` first.\n')
    exit()

items = pd.read_csv(items_path)
for col in items:
    if 'cloze' in col:
        del items[col]
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
    for i in range(0, len(keys), BATCH_SIZE):
        # Get batch
        _prompts = prompts[i:i+BATCH_SIZE].contiguous()
        _labels1 = labels1[i:i+BATCH_SIZE].contiguous()
        _labels3 = labels3[i:i+BATCH_SIZE].contiguous()
        _attention_mask = attention_mask[i:i+BATCH_SIZE]
        
        # Call on batch
        outputs = model(_prompts, attention_mask=_attention_mask)
        logits = outputs['logits'].contiguous()
        logits[:, :, tokenizer.pad_token_id] = -float("inf")
        preds = logits.argmax(axis=-1) * _attention_mask.int() # * (_labels1 >= 0).int()
        logits = logits.permute((0, 2, 1))

        # Compute surprisals
        _surprisals1 = torch.nn.CrossEntropyLoss(reduction='none')(logits, _labels1)
        _surprisals1 = np.asarray(_surprisals1.cpu())
        _surprisals1 = _surprisals1.sum(axis=-1)
        surprisals1.append(_surprisals1)

        _labels1[np.where(_labels1 < 0)] = 0
        _output_mask = (_labels1 >= 0).int()

#        print(_prompts)
#        print(_attention_mask)
#        print(logits)
#        print(_labels1)
#        print(_surprisals1)
#        for prompt, label, _preds in zip(_prompts, _labels1, preds):
#            print('"%s"' % tokenizer.decode(prompt))
#            print('"%s"' %tokenizer.decode(label))
#            print('"%s"' %tokenizer.decode(_preds))
#            print()
#        input()
        
        _surprisals3 = torch.nn.CrossEntropyLoss(reduction='none')(logits, _labels3)
        _surprisals3 = np.asarray(_surprisals3.cpu())
        _surprisals3 = _surprisals3.sum(axis=-1)
        surprisals3.append(_surprisals3)
        
surprisals1 = np.concatenate(surprisals1)
surprisals3 = np.concatenate(surprisals3)

items['gpt2'] = surprisals1
items['gpt2prob'] = np.exp(-items.gpt2)
items['gpt2region'] = surprisals3
items['gpt2regionprob'] = np.exp(-items.gpt2region)

items.to_csv(items_path, index=False)


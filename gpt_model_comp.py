import pandas as pd
import torch

import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config


validation_file = 'fine_tuning/datasets/sonnets.csv'

val_set = pd.read_csv(validation_file)
#val_set = val_set['text']


#for model_path in ["gpt2", "fine_tuning/sonnet_retrained_model", "fine_tuning/pft_model", "fine_tuning/twice_retrained"]:
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

val_set['tokens'] = [tokenizer(text, return_tensors="pt") for text in val_set['text']]

for model_path in ["gpt2", "fine_tuning/sonnet_retrained_model", "fine_tuning/pft_model", "fine_tuning/twice_retrained"]:

    if model_path == "gpt2":
        model = GPT2LMHeadModel.from_pretrained(model_path)
    else:
        config = GPT2Config.from_json_file(model_path + '/config.json')
        model = GPT2LMHeadModel.from_pretrained(model_path + '/pytorch_model.bin', config=config)

    model.eval()
    losses = []
    for step, tokens in enumerate(val_set['tokens']):
        with torch.no_grad():
            outputs = model(**tokens, labels=tokens['input_ids'])

        #print(outputs, dir(outputs), outputs.loss)
        loss = outputs.loss
        #losses.append(accelerator.gather(loss.repeat(per_device_eval_batch_size)))
        losses.append(loss)

    losses = torch.tensor(losses)
    #print(losses)
    print(model_path, ": mean: ", torch.mean(losses), ", var: ", torch.var(losses), ", perp: ", math.exp(torch.mean(losses)))


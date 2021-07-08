from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch
import random
import math
from torch.utils.data.dataloader import DataLoader
from datasets import load_dataset

from accelerate import Accelerator
from transformers import (
    default_data_collator,
)

import pandas as pd

print("loading")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

per_device_eval_batch_size = 8

#eval_dataset = pd.read_csv('fine_tuning/datasets/sonnets_validation.csv')
validation_file = 'fine_tuning/datasets/sonnets_validation.csv'

data_files = {}
data_files["validation"] = validation_file
extension = validation_file.split(".")[-1]
if extension == "txt":
    extension = "text"
raw_datasets = load_dataset(extension, data_files=data_files)

column_names = raw_datasets["validation"].column_names
text_column_name = "text" if "text" in column_names else column_names[0]

def tokenize_function(examples):
    return tokenizer(examples[text_column_name])

preprocessing_num_workers = None
overwrite_cache=False

tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    num_proc=preprocessing_num_workers,
    remove_columns=column_names,
    load_from_cache_file=not overwrite_cache,
    desc="Running tokenizer on dataset",
)

block_size = 1024

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=preprocessing_num_workers,
        load_from_cache_file=not overwrite_cache,
        desc=f"Grouping texts in chunks of 1",
    )

eval_dataset = lm_datasets["validation"]

eval_dataloader = DataLoader(
    eval_dataset, collate_fn=default_data_collator, batch_size=per_device_eval_batch_size
)


accelerator = Accelerator()

eval_dataloader = accelerator.prepare(eval_dataloader)

print("loading++")

for model_path in ["fine_tuning/sonnet_retrained_model", "fine_tuning/pft_model", "fine_tuning/twice_retrained"]:

    config = GPT2Config.from_json_file(model_path + '/config.json')
    model = GPT2LMHeadModel.from_pretrained(model_path + '/pytorch_model.bin', config=config)

    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(per_device_eval_batch_size)))

    losses = torch.cat(losses)
    losses = losses[: len(eval_dataset)]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    print(f"model {model_path}: perplexity: {perplexity}")


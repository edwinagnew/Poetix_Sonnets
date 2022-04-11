from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch
import random


"""
Things to change
"""
model_size = "custom"
model_path = "../fine_tuning/retrained_model_2"
seed = "From darkest forests"

selection_k = 5

n_tokens = 50
"""
end
"""



print("loading", model_size)

if model_size == "custom":
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    config = GPT2Config.from_json_file(model_path + '/config.json')
    model = GPT2LMHeadModel.from_pretrained(model_path + '/pytorch_model.bin', config=config)
else:
    tokenizer = GPT2Tokenizer.from_pretrained(model_size)
    model = GPT2LMHeadModel.from_pretrained(model_size)

if torch.cuda.is_available():
    print("putting to gpu")
    model.to('cuda')


print("loaded", model_size)



inputs = tokenizer(seed, return_tensors="pt")
context = inputs['input_ids']
generated = list(context[0])

past = None

for i in range(n_tokens):
    print(i)
    with torch.no_grad():
        outputs = model(context, past_key_values=past, use_cache=True)

    past = outputs.past_key_values
    output = outputs.logits

    while len(output) != 50257: output = output[0]


    top5 = torch.topk(output, selection_k)
    #print(top5, top5.indices)
    token = random.choice(top5.indices)
    #token = torch.argmax(output[..., -1, :])

    generated += [token.tolist()]
    context = token.unsqueeze(0)

sequence = tokenizer.decode(generated)

print(sequence)
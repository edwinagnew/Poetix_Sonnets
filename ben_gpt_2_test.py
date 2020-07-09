from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

if __name__ == '__main__':

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    model = GPT2LMHeadModel.from_pretrained('gpt2-large')
    generated = tokenizer.encode("The Manhattan bridge")
    context = torch.tensor([generated])
    past = None
    for i in range(100):
        print(i)
        output, past = model(context, past=past)
        token = torch.argmax(output[..., -1, :])
        generated += [token.tolist()]
        context = token.unsqueeze(0)
    sequence = tokenizer.decode(generated)
    print(sequence)
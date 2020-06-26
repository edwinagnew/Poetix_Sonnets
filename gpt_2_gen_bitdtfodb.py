import sonnet_basic
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
import random

from py_files import helper

class gpt:

    def __init__(self, seed, sonnet_object=None,  model="gpt2-large", template="FROM JJS NNS, PRPS VBP NN".split(), meter="0_10_10_1_01_01".split("_")):
        if sonnet_object:
            self.s = sonnet_basic
        else:
            self.s = sonnet_basic.Sonnet_Gen(mistakes_file=None)

        print("loading model")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model)
        self.model = GPT2LMHeadModel.from_pretrained(model)

        if seed: print(self.good_generation(seed, template, meter))

    def good_generation(self, seed, template="FROM JJS NNS, PRPS VBP NN".split(), meter="0_10_10_1_01_01".split("_"), b=6):
        """if not tokenizer or not model:
            print("loading model")
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2' + size)
            model = GPT2LMHeadModel.from_pretrained('gpt2' + size)
            print("loaded gpt2", size)"""

        if template and not meter: meter = [""] * len(template)
        words = list(self.tokenizer.encoder.keys())

        print("tokenizing")
        if not seed: seed = random.choice(self.s.get_pos_words(template[0], meter=meter[0])) #picks first word randomly
        generated = self.tokenizer.encode(seed)
        context = torch.tensor([generated])
        past = None

        punc = ",.;"

        if template:
            a, b = len(seed.split()), len(template)
        else:
            a, b = 0, b

        punc_next = False

        #for i in range(a, b):
        i = a
        while i < b:
            print(i)
            output, past = self.model(context, past=past)

            if template:
                output += abs(torch.min(output))  # makes all positive
                if punc_next:
                    poss = set(punc_next)
                    punc_next = False
                    #i -= 1
                else:
                    if template[i][-1] in punc:
                        template[i], punc_next = template[i][:-1], template[i][-1]
                    poss = set(self.s.get_pos_words(template[i], meter=meter[i]))  # searching a set is O(1), searching a list is O(n)!!!
                    print(template[i], meter[i], poss)
                # filt = torch.tensor([x.strip() in poss for x in tokenizer.encoder])
                # token = torch.argmax(output[..., -1, :][0] * filt)
                filt = np.array([int(x.strip("Ġ").lower() in poss) for x in self.tokenizer.encoder]) #"Ġ" is gpt-2's space character
                dist = helper.softmax(output[..., -1, :].detach().numpy() * filt, exclude_zeros=True)
                token = np.random.choice(np.arange(len(words)), p=dist).item()
                print("for ", template[i], end=': ')

            else:
                # token = torch.argmax(output[..., -1, :]).item()
                dist = helper.softmax(output[..., -1, :].detach().numpy())
                token = np.random.choice(np.arange(len(words)), p=dist).item()
            print("picked "+ str(token) + ": '"+ str(self.tokenizer.decode(token)) + "' with prob " + str(dist[token]))

            generated += [token]  # .tolist()
            # context = token.unsqueeze(0)
            context = torch.tensor(token).unsqueeze(0)

            i += int(not punc_next)

        print("tokens: ", generated)
        sequence = self.tokenizer.decode(generated)

        # while len(sequence.split()) != len(template):
        """for i in range(b):
            word_index = sequence.find(self.tokenizer.decode(generated[i]))
            if word_index - 1 > 0 and word_index - 1 != " " and sequence[word_index - 1] != "-":
                sequence = sequence[:word_index] + " " + sequence[word_index:]"""

        return sequence

    def gpt_2_score_line(self, line):
        input_ids = torch.tensor(self.tokenizer.encode(line, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = self.model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]
        return loss.item()
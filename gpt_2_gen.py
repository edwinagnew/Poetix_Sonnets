#import sonnet_basic
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
import random

from py_files import helper

class gpt:

    def __init__(self, seed, sonnet_method=None,  model="gpt2-large", template="FROM JJS NNS, PRPS VBP NN".split(), meter="0_10_10_1_01_01".split("_")):
        if sonnet_method:
            self.sonnet_words = sonnet_method
        else:
            input("didnt give me a sonnet_method, try again pls")
            #t = sonnet_basic.Sonnet_Gen()
            #self.sonnet_words = t.get_pos_words

        print("loading model")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model)
        self.model = GPT2LMHeadModel.from_pretrained(model)
        print("loaded", model)

        if seed: print(self.good_generation(seed, template, meter))

    def good_generation(self, seed=None, template="FROM JJS NNS, PRPS VBP NN".split(), meter="0_10_10_1_01_01".split("_"), rhyme_words=[], b=6, verbose=False):
        """if not tokenizer or not model:
            print("loading model")
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2' + size)
            model = GPT2LMHeadModel.from_pretrained('gpt2' + size)
            print("loaded gpt2", size)"""
        if type(template) != list: return self.good_generation(seed=seed, template=template.split(), meter=meter, rhyme_words=rhyme_words, verbose=verbose)
        if type(meter) != list: return self.good_generation(seed=seed, template=template, meter=meter.split("_"), rhyme_words=rhyme_words, verbose=verbose)
        if template and not meter: meter = [""] * len(template)
        words = list(self.tokenizer.encoder.keys())

        if verbose: print("tokenizing")
        if not seed: seed = random.choice(self.sonnet_words(template[0], meter=meter[0])) #picks first word randomly
        generated = self.tokenizer.encode(seed)
        context = torch.tensor([generated])
        past = None

        punc = ",.;?"

        if template:
            a, b = len(seed.split()), len(template)
        else:
            a, b = 0, b

        punc_next = False

        #for i in range(a, b):
        i = a
        while i < b:
            if verbose: print(i)
            output, past = self.model(context, past=past)

            if template:
                output += abs(torch.min(output))  # makes all positive
                if punc_next:
                    poss = set(punc_next)
                    punc_next = False
                    #i -= 1
                #elif template[i] == "POS":
                #    poss = {"'s"}
                else:
                    if template[i][-1] in punc:
                        template[i], punc_next = template[i][:-1], template[i][-1]
                    poss = set(self.sonnet_words(template[i], meter=meter[i]))  # searching a set is O(1), searching a list is O(n)!!!
                    if rhyme_words and i == b-1: poss = set(p for p in poss if p in rhyme_words)
                # filt = torch.tensor([x.strip() in poss for x in tokenizer.encoder])
                # token = torch.argmax(output[..., -1, :][0] * filt)
                if len(poss) == 1:
                    #choose token with right spacing
                    space = " " * int(list(poss)[0] not in punc + "'s")
                    token = self.tokenizer.encode(space + list(poss)[0])[0]
                    dist = np.ones(len(words))
                else:
                    filt = np.array([int(x.strip("Ġ").lower() in poss) for x in words]) #"Ġ" is gpt-2's space character
                    ws = output[..., -1, :].detach().numpy() * filt
                    dist = helper.softmax(ws, exclude_zeros=True)#, k=np.percentile(words, 0)) #TODO think about not necessarily softmaxing all words?
                    token = np.random.choice(np.arange(len(words)), p=dist).item()
                if verbose: print("for ", template[i], end=': ')

            else:
                # token = torch.argmax(output[..., -1, :]).item()
                dist = helper.softmax(output[..., -1, :].detach().numpy())
                token = np.random.choice(np.arange(len(words)), p=dist).item()

            if verbose: print("picked " + str(token) + ": '" + str(self.tokenizer.decode(token)) + "' with prob " + str(dist[token]))

            generated += [token]  # .tolist()
            # context = token.unsqueeze(0)
            context = torch.tensor(token).unsqueeze(0)

            i += int(not punc_next)

        if verbose: print("tokens: ", generated)
        sequence = self.tokenizer.decode(generated)

        # while len(sequence.split()) != len(template):
        """for i in range(b):
            word_index = sequence.find(self.tokenizer.decode(generated[i]))
            if word_index - 1 > 0 and word_index - 1 != " " and sequence[word_index - 1] != "-":
                sequence = sequence[:word_index] + " " + sequence[word_index:]"""

        return sequence

    def score_line(self, line):
        if type(line) == list: return [self.score_line(li.strip()) for li in line]
        input_ids = torch.tensor(self.tokenizer.encode(line, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = self.model(input_ids, labels=input_ids)
        #loss, logits = outputs[:2]
        return outputs[0].item()


#import sonnet_basic
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2DoubleHeadsModel
import torch
import numpy as np
import random
import string

from py_files import helper

class gpt_gen:

    def __init__(self, seed=None, sonnet_method=None,  model="gpt2-large", template="FROM JJS NNS, PRPS VBP NN".split(), meter="0_10_10_1_01_01".split("_")):
        if sonnet_method:
            self.sonnet_words = sonnet_method
        else:
            print("didnt give me a sonnet_method, wont be able to generate lines")
            #t = sonnet_basic.Sonnet_Gen()
            #self.sonnet_words = t.get_pos_words

        self.model_size = model

        print("loading model")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model)

        print("1")
        self.model = GPT2LMHeadModel.from_pretrained(model)
        self.mc_model = None
        print("loaded", model)



        if seed: print(self.good_generation(seed, template, meter))

    def good_generation(self, seed="", template="FROM JJS NNS, PRPS VBP NN".split(), meter="0_10_10_1_01_01".split("_"), rhyme_word=None, b=6, verbose=False):
        """if not tokenizer or not model:
            print("loading model")
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2' + size)
            model = GPT2LMHeadModel.from_pretrained('gpt2' + size)
            print("loaded gpt2", size)"""
        if type(template) != list: template = template.split()
        if type(meter) != list: meter = meter.split("_")
        if template and not meter: meter = [""] * len(template)
        words = list(self.tokenizer.encoder.keys())

        if template:
            #a, b = len(seed.split()), len(template) #complete partially started line
            a, b = 0, len(template) #write new line given previous ones
        else:
            a, b = 0, b

        if verbose: print("tokenizing")
        if not seed:
            seed = random.choice(self.sonnet_words(template[0], meter=meter[0])) #picks first word randomly
            a = 1
        generated = self.tokenizer.encode(seed)
        context = torch.tensor([generated])
        past = None

        punc = ",.;?"

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
                    if template[i][-1] == ">":
                        template[i], punc_next = template[i].split("<")[0], template[i].split("<")[-1].strip(">").split("/")

                    elif template[i][-1] in punc:
                        template[i], punc_next = template[i][:-1], template[i][-1]
                    poss = set(self.sonnet_words(template[i], meter=meter[i]))  # searching a set is O(1), searching a list is O(n)!!!
                    if i == b-1: poss = set(self.sonnet_words(template[i], meter=meter[i], rhyme=rhyme_word))
                    #if rhyme_words and i == b-1: poss = set(p for p in poss if p in rhyme_words)
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

        return sequence#.replace(seed, "")

    def score_line(self, line):
        if type(line) == list: return [self.score_line(li.strip()) for li in line]
        input_ids = torch.tensor(self.tokenizer.encode(line, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = self.model(input_ids, labels=input_ids)
        #loss, logits = outputs[:2]
        return outputs[0].item()

    def get_word_scores(self, line):
        if type(line) == list: return [self.get_word_scores(li.strip()) for li in line]
        input_ids = torch.tensor(self.tokenizer.encode(line, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = self.model(input_ids)
        return torch.tensor([outputs[0][0][max(0,i-1)][input_ids[0][i]] for i in range(len(input_ids[0]))]) #gets the score of each original word in the line

    def get_worst_suitable_word(self, template, meter, line, verbose=False, keep_last=True):
        words = line.replace("'s", " 's").translate(str.maketrans('', '', string.punctuation)).split()

        if verbose: print("line score before:", self.score_line(line))

        input_ids = torch.tensor(self.tokenizer.encode(line, add_special_tokens=True)).unsqueeze(0)
        if verbose: print([self.tokenizer.decode(t.item()) for t in input_ids[0]])

        word_scores = self.get_word_scores(line)
        #if word_scores.min() < 0: word_scores -= word_scores.min()
        if keep_last:
            if line[-1] in string.punctuation:
                word_scores[-2] = 0
            else:
                word_scores[-1] = 0

        worst_token_index = -1
        worst_index = -1
        worst_word = ""
        while worst_word not in words or worst_index < 1 or len(self.sonnet_words(template[worst_index], meter[worst_index])) < 2:
        #while self.tokenizer.decode(input_ids[0][worst_token_index].item()).strip() not in words or worst_index < 1 or len(self.sonnet_words(template[worst_index], meter[worst_index])) < 2:
            if verbose and worst_token_index > -1: print("changing", self.tokenizer.decode(input_ids[0][worst_token_index].item()))
            if worst_token_index + worst_index >= 0: word_scores[worst_token_index] = 0
            worst_token_index = torch.argmin(word_scores).item()
            worst_word = self.tokenizer.decode(input_ids[0][worst_token_index].item()).strip()
            if worst_word in words: worst_index = words.index(worst_word)
            if word_scores.min() == 0:
                print("couldnt find any word to change")
                return None, None
        if verbose: print("word scores:", word_scores.tolist())
        if verbose: print("worst", worst_token_index)
        if verbose: print("worst word was ", line.split()[worst_index], worst_word)

        return worst_index, worst_token_index

    def iterative_improve_loss(self, template, meter, line, selection="random", n=1, k=5, verbose=False, keep_rhyme_word=True):
        if n == 0: return line
        if type(template) != list: template = template.split()
        if type(meter) != list: meter = meter.split("_")

        input_ids = torch.tensor(self.tokenizer.encode(line, add_special_tokens=True)).unsqueeze(0)

        worst_index, worst_token_index = self.get_worst_suitable_word(template, meter, line, verbose=verbose, keep_last=keep_rhyme_word)
        if not worst_index: return line

        # random
        if selection == "random":
            choices = []
            scores = []
            poss = self.sonnet_words(helper.remove_punc(template[worst_index]), meter[worst_index])
            for i in range(k):
                if verbose: print(i, "getting ", helper.remove_punc(template[worst_index]), meter[worst_index])
                new_word = random.choice(poss)
                new_line = input_ids[0].tolist()
                new_line = new_line[:worst_token_index] + self.tokenizer.encode(" " + new_word) + new_line[worst_token_index + 1:]
                choices.append(self.tokenizer.decode(new_line))
                scores.append(self.score_line(choices[-1]))

        # top k
        else:
            poss = self.sonnet_words(template[worst_index].translate(str.maketrans('', '', string.punctuation)), meter[worst_index])
            words = list(self.tokenizer.encoder.keys())
            filt = np.array([int(x.strip("Ġ").lower() in poss) for x in words])
            if verbose: print("filtering:", self.tokenizer.decode([x for x in range(len(words)) if filt[x]]), sum(filt))
            output, past = self.model(input_ids)
            output += abs(output.min())
            out = torch.tensor(output[..., worst_token_index - 1, :].detach().numpy() * filt)  # predicts what comes after token, not whats there?

            best_vals, best_is = out.topk(k)

            if verbose: print("top k: ", best_vals, best_is)

            choices = []
            scores = []
            for b in best_is[0]:
                new_line = input_ids[0].tolist()
                new_line[worst_token_index] = b.item()
                choices.append(self.tokenizer.decode(new_line))
                scores.append(self.score_line(choices[-1]))

        if verbose: print("choices", choices)
        if verbose: print("scores", scores)

        best_line = choices[np.argmin(scores)]
        print(n, "best =", best_line, min(scores))

        return self.iterative_improve_loss(template, meter, best_line, n=n-1, k=k, verbose=verbose)

    def iterative_improve_mc(self, template, meter, line, n=1, k=5, selection="random", verbose=False):
        if n == 0: return line
        if type(template) != list: template = template.split()
        if type(meter) != list: meter = meter.split("_")

        input_ids = torch.tensor(self.tokenizer.encode(line, add_special_tokens=True)).unsqueeze(0)

        worst_index, worst_token_index = self.get_worst_suitable_word(template, meter, line, verbose=verbose)
        if not worst_index: return line

        #assert self.tokenizer.decode(input_ids[0][worst_token_index].item()).strip() == line.split()[worst_index]

        self.load_mc_model()

        # random options
        if selection == "random":
            choices = []
            for i in range(k):
                #assuming there's no punctuation
                if verbose: print(i, "getting ", template[worst_index], meter[worst_index])
                new_word = random.choice(self.sonnet_words(template[worst_index].translate(str.maketrans('', '', string.punctuation)), meter[worst_index]))
                new_line = input_ids[0].tolist()
                #new_line[worst_token_index] = self.mc_tokenizer.encode(" " + new_word)[0]
                new_line = new_line[:worst_token_index] + self.mc_tokenizer.encode(" " + new_word) + new_line[worst_token_index + 1:]
                new_line += self.mc_tokenizer.encode('[CLS]')
                #new_line = " ".join(new_line) + ' [CLS]'
                choices.append(new_line)

        else:
            #top k options
            choices = []
            poss = self.sonnet_words(helper.remove_punc(template[worst_index]), meter[worst_index])
            words = list(self.tokenizer.encoder.keys())
            filt = np.array([int(x.strip("Ġ").lower() in poss) for x in words])
            output, past = self.model(input_ids)
            output += abs(output.min())
            out = torch.tensor(output[..., worst_token_index - 1, :].detach().numpy() * filt)  # predicts what comes after token, not whats there?

            best_vals, best_is = out.topk(k)

            for b in best_is[0]:
                new_line = input_ids[0].tolist()
                new_line[worst_token_index] = b.item()
                new_line += self.mc_tokenizer.encode('[CLS]')
                choices.append(new_line)

        if verbose: print([self.mc_tokenizer.decode(c) for c in choices], choices)

        choice_scores = self.multiple_choice(choices)

        if verbose: print(choice_scores)

        best_line = self.mc_tokenizer.decode(choices[torch.argmax(choice_scores).item()]).replace(" [CLS]", "")

        print(n, "best = ", best_line, self.score_line(best_line))

        return self.iterative_improve_mc(template, meter, best_line, n=n-1, k=k, verbose=verbose)


    def multiple_choice(self, choices):
        cls_token_location = [tokens.index(self.mc_tokenizer.cls_token_id) for tokens in choices]
        input_ids = torch.tensor(choices).unsqueeze(0)  # Batch size: 1, number of choices: 2
        mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1

        outputs = self.mc_model(input_ids, mc_token_ids=mc_token_ids)
        lm_prediction_scores, mc_prediction_scores = outputs[:2]

        return mc_prediction_scores

    def load_mc_model(self):
        if not self.mc_model:
            print("loading mc_model")
            self.mc_tokenizer = GPT2Tokenizer.from_pretrained(self.model_size)
            num_added_tokens = self.mc_tokenizer.add_special_tokens({'cls_token': '[CLS]'})
            self.mc_model = GPT2DoubleHeadsModel.from_pretrained(self.model_size)
            embedding_layer = self.mc_model.resize_token_embeddings(len(self.mc_tokenizer))








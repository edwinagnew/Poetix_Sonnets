import pickle
import random
import torch
import json
import numpy as np
import pandas as pd

from py_files import helper

from transformers import BertTokenizer, BertForMaskedLM
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import pronouncing

from nltk.corpus import wordnet as wn
from nltk import PorterStemmer


from pattern.en import comparative, superlative, pluralize



class Scenery_Gen():
    def __init__(self, model="bert", postag_file='saved_objects/postag_dict_all+VBN.p',
                 syllables_file='saved_objects/cmudict-0.7b.txt',
                 extra_stress_file='saved_objects/edwins_extra_stresses.txt',
                 top_file='saved_objects/words/top_words.txt',
                 templates_file='poems/scenery_templates.txt',
                 mistakes_file=None):

        with open(templates_file) as tf:
            self.templates = [(" ".join(line.split()[:-1]), line.split()[-1]) for line in tf.readlines()]
        with open(postag_file, 'rb') as f:
            self.postag_dict = pickle.load(f)
        self.pos_to_words, self.words_to_pos = helper.get_pos_dict(postag_file, mistakes_file=mistakes_file)


        self.special_words = helper.get_finer_pos_words()

        self.dict_meters = helper.create_syll_dict([syllables_file], extra_stress_file)

        if model == "bert":
            self.lang_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.lang_vocab = list(self.tokenizer.vocab.keys())
            self.lang_model.eval()
            self.vocab_to_num = self.tokenizer.vocab

        elif model == "roberta":
            self.lang_model = RobertaForMaskedLM.from_pretrained('roberta-base') # 'roberta-base'
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base') # 'roberta-large'
            with open("saved_objects/roberta/vocab.json") as json_file:
                j = json.load(json_file)
            self.lang_vocab = list(j.keys())
            self.lang_model.eval()
            self.vocab_to_num = {self.lang_vocab[x]: x for x in range(len(self.lang_vocab))}

        elif model == "gpt_2":
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')

        else:
            self.lang_model = None

        self.poems = pd.read_csv('poems/kaggle_poem_dataset.csv')['Content']

        with open(top_file) as tf:
            self.top_common_words = [line.strip() for line in tf.readlines()][:125]

        self.stemmer = PorterStemmer()

        self.api_url = 'https://api.datamuse.com/words'

        self.gender = random.choice([["he", "him", "his", "himself"], ["she", "her", "hers", "herself"]])


    def fix_line(self, words, word_number, POS, meter, verbose = False):
        potential_verbs = self.get_pos_words(POS, meter)
        best_score = float("inf")
        best_word = ""
        for verb in potential_verbs:
            new_line = words
            new_line[word_number] = verb
            input_ids = torch.tensor(self.tokenizer.encode(" ".join(new_line), add_special_tokens=False)).unsqueeze(0)  # tokenizes
            tokens = [self.lang_vocab[x] for x in input_ids[0]]
            loss, outputs = self.lang_model(input_ids, masked_lm_labels=input_ids)  # masks each token and gives probability for all tokens in each word. Shape num_words * vocab_size
            softmax = torch.nn.Softmax(dim=1)  # normalizes the probabilites to be between 0 and 1
            outputs = softmax(outputs[0])
            extra_token = ""
        # for word_number in range(0,len(line)-1): #ignore  last word to keep rhyme
            k = tokens.index(self.tokenizer.tokenize(line[-1])[0])  # where the last word begins

            out_number = word_number

            if loss.item() < best_score:
                best_score = loss
                best_word = verb

        print(best_word)
        line[word_number] = self.lang_vocab[np.argmax(outputs[out_number].detach().numpy())]

    def gpt_2_score_line(self, line):
        input_ids = torch.tensor(self.tokenizer.encode(line, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs =self.model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]
        return loss.item()

    def score_line(self, words):
        line = words.split()

        input_ids = torch.tensor(self.tokenizer.encode(" ".join(line), add_special_tokens=False)).unsqueeze(0)  # tokenizes
        loss, outputs = self.lang_model(input_ids,masked_lm_labels=input_ids)  # masks each token and gives probability for all tokens in each word. Shape num_words * vocab_size
        return loss.item()


    def get_word_pos(self, word):
        """
        Get the set of POS category of a word. If we are unable to get the category, return None.
        """
        # Special case
        if word.upper() in self.special_words:
            return [word.upper()]
        if word not in self.words_to_pos:
            return []
        return self.words_to_pos[word]

    def get_pos_words(self, pos, meter=None):
        """
        Gets all the words of a given POS
        Parameters
        ----------
        pos - the POS you want
        meter - (optional) returns only words which fit the given meter, e.g. 101
        """
        if pos in self.special_words:
            return [pos.lower()]
        if "PRP" in pos:
            ret = [p for p in self.pos_to_words[pos] if p in self.gender]
            if len(ret) == 0: input(pos + meter + str(self.gender) + str([self.dict_meters[p] for p in self.gender]))
            return ret
        if pos not in self.pos_to_words:
            return None
        if meter:
            ret = [word for word in self.pos_to_words[pos] if word in self.dict_meters and meter in self.dict_meters[word]]
            if len(ret) == 0:
                return False
            return ret
        return self.pos_to_words[pos]

    def get_meter(self, word):
        if word not in self.dict_meters: return []
        return self.dict_meters[word]

    def suitable_last_word(self, word, line):
        pos = self.templates[line][0].split()[-1].split("sc")[-1]
        meter = self.templates[line][1].split("_")[-1]
        return pos in self.get_word_pos(word) and meter in self.dict_meters[word]

    def write_stanza(self, theme="forest", verbose=True):
        """
        Possible approach :
        1. Write a preliminary line
            a. fits template and meter randomly and make sure end word has at least 20(?) rhyming words
            b. inserts thematic words where possible(?)
        2. Use bert (or roberta) to change every word (several times?) with weighted probabilities it gives, filtered for meter and template and perhaps relevant words boosted?

        Forest:
        1. forest (or synonyms)
        2. branches
        3. roots
        4. leaves
        -------

        """
        """rhyme_dict = {}
        for i in ['A', 'B']:
            rhyme_dict[i] = self.getRhymes([theme], words=self.words_to_pos)
        last_word_dict = self.last_word_dict(rhyme_dict)"""

        theme_words = self.get_theme_words(theme, verbose=False)
        print(theme_words)
        lines = []
        orig_lines = []
        for line_number, (template, meter) in enumerate(self.templates):
            if line_number % 4 == 0: rhymes = []
            template = template.split()
            meter = meter.split("_")
            line = ""
            for i in range(len(template) - 1):
                new_words = []
                scores = []
                pos = template[i]
                if "sc" in template[i]:
                    pos = template[i].split("sc")[-1]
                    for thematic in theme_words:
                        #print(thematic)
                        if theme_words[thematic] > 1 and meter[i] in self.dict_meters[thematic] and pos in self.get_word_pos(thematic) :
                            new_words.append(thematic)
                            scores.append(theme_words[thematic])
                            #if verbose: print("found ", thematic, theme_words[thematic], "for ", meter[i], template[i])
                if len(new_words) == 0:
                    new_word = random.choice(self.get_pos_words(pos, meter=meter[i]))
                else:
                    dist = helper.softmax(scores)
                    new_word = np.random.choice(new_words, p=dist)
                    theme_words[new_word] /= 2 #don't choose same word twice
                    theme_words[self.stemmer.stem(new_word)] = 0
                    if "NN" in self.get_word_pos(new_word): theme_words[pluralize(new_word)] = 0
                    print(new_word, " chosen with prob", dist[new_words.index(new_word)])
                line += new_word + " "

            if line_number % 4 < 2:
                word = None
                #high_score = 0
                rhyme_pos = self.templates[min(13, line_number + 2)][0].split()[-1].split("sc")[-1]
                rhyme_met = self.templates[min(13, line_number + 2)][1].split("_")[-1]
                while not word or not any(r in self.get_pos_words(rhyme_pos, meter=rhyme_met) for r in pronouncing.rhymes(word)):
                    word = random.choice(self.get_pos_words(template[-1].split("sc")[-1], meter=meter[-1]))
                line += word
                rhymes.append(word)
            else:
                n = -2
                if line_number == 13: n = -1
                line += random.choice([rhyme for rhyme in self.get_pos_words(template[-1].split("sc")[-1], meter=meter[-1]) if self.rhymes(rhyme, lines[n].split()[-1])])
            #line += random.choice(last_word_dict[line_number])
            if self.lang_model:
                print("line initially ", line)
                orig_lines.append(line)
                rhyme_set = []
                for r in rhymes:
                    rhyme_set += pronouncing.rhymes(r)
                print("rhyme_set length: ", len(rhyme_set))
                line = self.update_bert(line.strip().split(), meter, template, len(template)/2, theme_words=theme_words, rhyme_words=rhyme_set, filter_meter=True, verbose=True)
            print("line now ", line)
            lines.append(line)
            #break
        print("")
        [print(orig_lines[i]) for i in range(len(orig_lines))]
        print("")
        print("         ---", theme.upper(), "---       ")
        for cand in range(len(lines)):
            print(lines[cand])
            if ((cand + 1) % 4 == 0): print("")

    def update_bert(self, line, meter, template, iterations, theme_words=[], rhyme_words=[], filter_meter=True, verbose=False, choice = "min"):
        if iterations <= 0: return " ".join(line) #base case
        #TODO deal with tags like ### (which are responsible for actually cool words)
        input_ids = torch.tensor(self.tokenizer.encode(" ".join(line), add_special_tokens=False)).unsqueeze(0) #tokenizes
        tokens = [self.lang_vocab[x] for x in input_ids[0]]
        loss, outputs = self.lang_model(input_ids, masked_lm_labels=input_ids) #masks each token and gives probability for all tokens in each word. Shape num_words * vocab_size
        if verbose: print("loss = ", loss)
        softmax = torch.nn.Softmax(dim=1) #normalizes the probabilites to be between 0 and 1
        outputs = softmax(outputs[0])
        extra_token = ""
        #for word_number in range(0,len(line)-1): #ignore  last word to keep rhyme
        k = tokens.index(self.tokenizer.tokenize(line[-1])[0])  # where the last word begins

        if choice == "rand":
            word_number = out_number = random.choice(np.arange(k))

        elif choice == "min":
            probs = np.array([outputs[i][self.vocab_to_num[tokens[i]]] for i in range(k)])
            word_number = out_number = np.argmin(probs)
            while tokens[out_number].upper() in self.special_words or any(x in self.get_word_pos(tokens[out_number]) for x in ["PRP", "PRP$"]):
                if verbose: print("least likely is unchangable ", tokens, out_number, outputs[out_number][input_ids[0][out_number]])
                probs[out_number] *= 10
                word_number = out_number = np.argmin(probs)

        if len(outputs) > len(line):
            if verbose: print("before: predicting", self.lang_vocab[input_ids[0][out_number]], tokens)
            if tokens[out_number] in line:
                word_number = line.index(tokens[out_number])
                t = 1
                while "##" in self.lang_vocab[input_ids[0][out_number + t]]:
                    extra_token += self.lang_vocab[input_ids[0][out_number + 1]].split("#")[-1]
                    t += 1
                    if out_number + t >= len(input_ids[0]):
                        if verbose: print("last word chosen --> restarting", 1/0)
                        return self.update_bert(line, meter, template, iterations, theme_words=theme_words, rhyme_words=rhyme_words, verbose=verbose)
            else:
                sub_tokens = [self.tokenizer.tokenize(w)[0] for w in line]
                while self.lang_vocab[input_ids[0][out_number]] not in sub_tokens: out_number -= 1
                word_number = sub_tokens.index(self.lang_vocab[input_ids[0][out_number]])
                t = 1
                while "##" in self.lang_vocab[input_ids[0][out_number + t]]:
                    extra_token += self.lang_vocab[input_ids[0][word_number + t]].split("#")[-1]
                    t += 1
                    if out_number + t >= len(input_ids[0]):
                        if verbose: print("last word chosen --> restarting", 1/0)
                        return self.update_bert(line, meter, template, iterations, theme_words=theme_words, rhyme_words=rhyme_words, verbose=verbose)

            if verbose: print("after: ", out_number, word_number, line, " '", extra_token, "' ")

        #if verbose: print("word number ", word_number, line[word_number], template[word_number], "outnumber:", out_number)

        temp = template[word_number].split("sc")[-1]
        if len(self.get_pos_words(temp)) > 1 and temp not in ['PRP', 'PRP$']: #only change one word each time?
            filt = np.array([int( temp in self.get_word_pos(word) or temp in self.get_word_pos(word + extra_token)) for word in self.lang_vocab])
            if filter_meter and meter: filt *= np.array([int(meter[word_number] in self.get_meter(word) or meter[word_number] in self.get_meter(word + extra_token)) for word in self.lang_vocab])
            predictions = outputs[out_number].detach().numpy() * filt #filters non-words and words which dont fit meter and template

            for p in range(len(predictions)):
                if predictions[p] > 0.001 and self.lang_vocab[p] in rhyme_words:
                    print("weighting internal rhyme '", self.lang_vocab[p], "', orig: ", predictions[p], ", now: ", predictions[p]*5/sum(predictions))
                    predictions[p] *= 5
                if predictions[p] > 0.001 and self.lang_vocab[p] in theme_words and "sc" in template[word_number]:
                    b = predictions[p]
                    predictions[p] *= theme_words[self.lang_vocab[p]]**2
                    if verbose: print("weighting thematic '", self.lang_vocab[p], "' by ", theme_words[self.lang_vocab[p]], ", now: ", predictions[p]/sum(predictions), ", was: ", b)

            predictions /= sum(predictions)
            if verbose: print("predicting a ", template[word_number], meter[word_number], " for ", word_number, ". min: ", min(predictions), " max: ", max(predictions), "sum: ", sum(predictions), ", ", {self.lang_vocab[p]: predictions[p] for p in range(len(predictions)) if predictions[p] > 0})

            if iterations > 1:
                line[word_number] = np.random.choice(self.lang_vocab, p=predictions)
            else: #greedy for last iteration
                line[word_number] = self.lang_vocab[np.argmax(predictions)]

            print("word now ", line[word_number], "prob: ", predictions[self.lang_vocab.index(line[word_number])])

            if temp not in self.get_word_pos(line[word_number]):
                line[word_number] += extra_token
                if temp not in self.get_word_pos(line[word_number]): #still not valid
                    print("Extra token didnt help ", template[word_number], line[word_number], extra_token)
                    print(1/0)


            if verbose: print("line now", line)
        else:
            if verbose: print("picked ", line[word_number], "which is bad word")
            iterations += 1
        return self.update_bert(line, meter, template, iterations-1, theme_words=theme_words, rhyme_words=rhyme_words, verbose=verbose)

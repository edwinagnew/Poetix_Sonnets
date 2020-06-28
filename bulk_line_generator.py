import pickle
import random
import torch
import json
import numpy as np
import string
import pandas as pd

from py_files import helper
import theme_word_file

from transformers import BertTokenizer, BertForMaskedLM
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import pronouncing

from nltk.corpus import wordnet as wn
from nltk import PorterStemmer

import poem_core

class Bulk_Gen(poem_core.Poem):
    def __init__(self, model=None, postag_file='saved_objects/postag_dict_all+VBN.p',
                 syllables_file='saved_objects/cmudict-0.7b.txt',
                 extra_stress_file='saved_objects/edwins_extra_stresses.txt',
                 top_file='saved_objects/words/top_words.txt',
                 templates_file="poems/jordan_templates.txt",
                 paired_templates=False,
                 #templates_file='poems/number_templates.txt',
                 mistakes_file='saved_objects/mistakes.txt'):

        #self.templates = [("FROM scJJS scNNS PRP VBZ NN", "0_10_10_1_01_01"),
         #                 ("THAT scJJ scNN PRP VBD MIGHT RB VB", "0_10_10_1_0_10_1"),
          #                ("WHERE ALL THE scNNS OF PRP$ JJ NNS", "0_1_0_10_1_0_10_1"),
           #               ("AND THAT JJ WHICH RB VBZ NN", "0_1_01_0_10_1_01")]

        poem_core.Poem.__init__(self, words_file="saved_objects/tagged_words.p", templates_file=templates_file,
                                syllables_file=syllables_file, extra_stress_file=extra_stress_file, top_file=top_file)
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

        if paired_templates == True:
            self.temp_pairs = {}
            for i in range(len(self.templates)):
                if i % 2 == 1:
                    self.temp_pairs[self.templates[i-1]] = self.templates[i]


        #with open('poems/kaggle_poem_dataset.csv', newline='') as csvfile:
         #   self.poems = csv.DictReader(csvfile)
        self.poems = list(pd.read_csv('poems/kaggle_poem_dataset.csv')['Content'])
        self.surrounding_words = {}

        self.gender = random.choice([["he", "him", "his", "himself"], ["she", "her", "hers", "herself"]])

    #override
    def get_pos_words(self,pos, meter=None, phrase=()):
        """
        Gets all the words of a given POS
        Parameters
        ----------
        pos - the POS you want
        meter - (optional) returns only words which fit the given meter, e.g. 101
        phrase (optional) - returns only words which have a phrase in the dataset. in format ([word1, word2, word3], i) where i is the index of the word to change since the length can be 2 or 3
        """
        punc = [".", ",", ";", "?", ">"]
        if pos[-1] in punc:
            p = pos[-1]
            if p == ">":
                p = random.choice(pos.split("<")[-1].strip(">").split("/"))
                pos = pos.split("<")[0] + p
            return [word + p for word in self.get_pos_words(pos[:-1], meter=meter)]
        if pos in self.special_words:
            return [pos.lower()]
        if "PRP" in pos and "_" not in pos and meter:
            ret = [p for p in self.pos_to_words[pos] if
                   p in self.gender and any(len(meter) == len(q) for q in self.get_meter(p))]
            if len(ret) == 0: ret = [input("PRP not happening " + pos + " '" + meter + "' " + str(self.gender) + str([self.dict_meters[p] for p in self.gender]))]
            return ret
        elif pos not in self.pos_to_words:
            return []
        if meter:
            ret = [word for word in self.pos_to_words[pos] if
                   word in self.dict_meters and meter in self.dict_meters[word]]
            return ret
        return [p for p in self.pos_to_words[pos]]


    #@override
    def last_word_dict(self, rhyme_dict):
        """
        Given the rhyme sets, extract all possible last words from the rhyme set
        dictionaries.

        Parameters
        ----------
        rhyme_dict: dictionary
            Format is   {'A': {tone1 : {similar word: [rhyming words], similar word: [rhyming words], etc.}}, {tone2:{...}}},
                        'B': {tone1 : {similar word: [rhyming words], similar word: [rhyming words], etc.}}, {tone2:{...}}}
                        etc
        Returns
        -------
        dictionary
            Format is {1: ['apple', 'orange'], 2: ['apple', orange] ... }

        """
        scheme = {1: 'A', 2: 'B', 3: 'A', 4: 'B'}
        last_word_dict = {}

        first_rhymes = []
        for i in range(1, len(scheme) + 1):
            if i in [1, 2]:  # lines with a new rhyme -> pick a random key
                last_word_dict[i] = [random.choice(list(rhyme_dict[scheme[i]].keys()))]  # NB ensure it doesnt pick the same as another one
                j = 0
                while not self.suitable_last_word(last_word_dict[i][0], i - 1) or last_word_dict[i][0] in first_rhymes:
                    # or any(rhyme_dict['A'][last_word_dict[i][0]] in rhyme_dict['A'][word] for word in first_rhymes):
                    last_word_dict[i] = [random.choice(list(rhyme_dict[scheme[i]].keys()))]
                    if not any(self.templates[i - 1][1].split("_")[-1] in self.dict_meters[w] for w in
                               rhyme_dict[scheme[i]]):
                        word = last_word_dict[i][0]
                        if self.templates[i - 1][0].split()[-1] in self.get_word_pos(word) and len(
                                self.dict_meters[word][0]) == len(self.templates[i - 1][1].split("_")[-1]) and any(
                                self.suitable_last_word(r, i + 1) for r in rhyme_dict[scheme[i]][word]):
                            self.dict_meters[word].append(self.templates[i - 1][1].split("_")[-1])
                            print("cheated with ", word, " ", self.dict_meters[word],
                                  self.suitable_last_word(word, i - 1))
                    j += 1
                    if j > len(rhyme_dict[scheme[i]]) * 2: input(str(scheme[i]) + " " + str(rhyme_dict[scheme[i]]))
                first_rhymes.append(last_word_dict[i][0])

            if i in [3, 4]:  # lines with an old rhyme -> pick a random value corresponding to key of rhyming couplet
                letter = scheme[i]
                pair = last_word_dict[i - 2][0]
                last_word_dict[i] = [word for word in rhyme_dict[letter][pair] if self.suitable_last_word(word, i - 1)]
                if len(last_word_dict[i]) == 0:
                    print("fuck me", last_word_dict, i, self.templates[i])
                    print(1 / 0)
        return last_word_dict

    #@ovveride
    def suitable_last_word(self, word, line):
        pos = self.templates[line][0].split()[-1].split("sc")[-1]
        meter = self.templates[line][1].split("_")[-1]
        return pos in self.get_word_pos(word) and meter in self.dict_meters[word]

    def write_bulk_lines(self, num_lines, theme="flower"):
        """
        Writes a poem from the templates
        Parameters
        ----------
        theme - what the theme words should be about
        checks - a list of POS to check that a phrase exists in the corpus

        Returns
        -------

        """
        """rhyme_dict = {}
        for i in ['A', 'B']:
            rhyme_dict[i] = self.getRhymes([theme], words=self.words_to_pos)
        last_word_dict = self.last_word_dict(rhyme_dict)"""
        theme_gen = theme_word_file.Theme()
        theme_words = theme_gen.get_theme_words(theme, verbose=False)
        lines = []

        for i in range(num_lines):
            self.reset_number_words()
            template, meter = random.choice(self.templates)
            template = template.split()
            meter = meter.split("_")
            line = self.write_line(template, meter, theme_words=theme_words)
            if line:
                lines.append(line)
            """
            for check in checks:
                if check in template:
                    adv = template.index(check)
                    line_arr = line.split()
                    #phrase = []
                    low = max(0,adv-1)
                    if template[low] in self.special_words: low += 1
                    #phrase.append(line_arr[low])
                    #phrase.append(line_arr[adv])
                    high = min(len(line_arr), adv+2)
                    if template[high-1] in self.special_words: high -= 1
                    #phrase.append(line_arr[high])
                    print(check, " adv ", line, "low: ", low, ", high: ", high, "adv: ", adv, line_arr[low:high])
                    inc_syn = False
                    while not self.phrase_in_poem_fast(line_arr[low:high], include_syns=inc_syn):
                        poss = self.get_pos_words(check, meter=meter[adv], phrase=(line_arr[low:high], line_arr[low:high].index(line_arr[adv])))
                        inc_syn = True
                        if not poss:
                            continue
                        else:
                            line = self.write_line(line_number, template, meter, theme_words=theme_words)
                            inc_syn = False
                            continue
                            
                        line_arr[adv] = random.choice(poss)
                        print(check, " updated, line now ", " ".join(line_arr))
                    line = " ".join(line_arr)
                    
        for cand in range(len(lines)):
            print(lines[cand])
            if ((cand + 1) % 4 == 0): print("")
        """
        return lines

    def write_line(self, template, meter, theme_words=[], verbose=True):
        #numbers = {}
        line = ""
        for i in range(len(template) - 1):
            p = ""
            new_words = []
            scores = []
            pos = template[i]
            punc = [".", ",", ";", "?", ">"]
            if pos[-1] in punc:
                p = pos[-1]
                if p == ">":
                    p = random.choice(pos.split("<")[-1].strip(">").split("/"))
                    pos = pos.split("<")[0] + p
                pos = pos[:-1]
            #num = -1
            #letter = ""
            """if "_" in pos:
                print("b")
                try: num = int(pos.split("_")[1])
                except: letter = pos.split("_")[1]
                pos = pos.split("_")[0] """
            if "sc" in pos:
                print("c")
                pos = pos.split("sc")[-1]
                if pos not in self.pos_to_words: input("need to implement scenery rhetorical words")
                for thematic in theme_words[pos]:
                    if theme_words[pos][thematic] > 0.1 and meter[i] in self.dict_meters[thematic]:
                        new_words.append(thematic)
                        scores.append(theme_words[pos][thematic])
            if len(new_words) == 0:
                #new_word = random.choice(self.get_pos_words(pos, meter=meter[i]))
                poss = self.get_pos_words(pos, meter=meter[i])
                if not poss:
                    return False
                """if num in numbers:
                    print("checking", numbers[num], ":", poss)
                    dist = [helper.get_spacy_similarity(numbers[num], w) for w in poss]
                    new_word = np.random.choice(poss, p=helper.softmax(dist))
                    if verbose: print("weighted choice of ", new_word, ", related to ", numbers[num], "with prob ", dist[poss.index(new_word)])"""

                if len(poss) == 1: new_word = poss[0]

                else: new_word = np.random.choice(poss, p=self.my_softmax([self.pos_to_words[pos][w] for w in poss])) + p #softmax loses distinctions if any?
            else:
                dist = self.my_softmax(scores)
                new_word = np.random.choice(new_words, p=dist) + p
                theme_words[pos][new_word] = theme_words[pos][new_word] / 4.0  # don't choose same word twice
                #theme_words[self.stemmer.stem(new_word)] = 0
                #if "NN" in self.get_word_pos(new_word): theme_words[pluralize(new_word)] = 0
                if verbose: print(new_word, " chosen with prob", dist[new_words.index(new_word)], "now: ", theme_words[pos][new_word])
            line += new_word + " "

        pos = template[-1].split("sc")[-1]
            #num = -1
            #letter = ""
        if len(self.get_pos_words(pos, meter=meter[-1])) == 0:
            return False
        word = random.choice(self.get_pos_words(pos, meter=meter[-1]))
        line += word
        return line

    def write_line_pairs(self,  num_lines, theme="flower"):

        theme_gen = theme_word_file.Theme()
        theme_words = theme_gen.get_theme_words(theme, verbose=False)
        lines = []
        next_temp = False
        for i in range(num_lines):
            self.reset_number_words()
            if next_temp == False:
                template, meter = random.choice(list(self.temp_pairs.keys()))
                next_temp = self.temp_pairs[(template, meter)]
            else:
                template, meter = next_temp
                next_temp = False
            template = template.split()
            meter = meter.split("_")
            line = self.write_line(template, meter, theme_words=theme_words)
            if line:
                lines.append(line)
            """
            for check in checks:
                if check in template:
                    adv = template.index(check)
                    line_arr = line.split()
                    #phrase = []
                    low = max(0,adv-1)
                    if template[low] in self.special_words: low += 1
                    #phrase.append(line_arr[low])
                    #phrase.append(line_arr[adv])
                    high = min(len(line_arr), adv+2)
                    if template[high-1] in self.special_words: high -= 1
                    #phrase.append(line_arr[high])
                    print(check, " adv ", line, "low: ", low, ", high: ", high, "adv: ", adv, line_arr[low:high])
                    inc_syn = False
                    while not self.phrase_in_poem_fast(line_arr[low:high], include_syns=inc_syn):
                        poss = self.get_pos_words(check, meter=meter[adv], phrase=(line_arr[low:high], line_arr[low:high].index(line_arr[adv])))
                        inc_syn = True
                        if not poss:
                            continue
                        else:
                            line = self.write_line(line_number, template, meter, theme_words=theme_words)
                            inc_syn = False
                            continue

                        line_arr[adv] = random.choice(poss)
                        print(check, " updated, line now ", " ".join(line_arr))
                    line = " ".join(line_arr)

        for cand in range(len(lines)):
            print(lines[cand])
            if ((cand + 1) % 4 == 0): print("")
        """
        return lines



    def rhymes(self, word1, word2):
        return word1 in pronouncing.rhymes(word2) or word2 in pronouncing.rhymes(word1)

    def phrase_in_poem(self, words, ret_lines=False, include_syns=False):
        """
        Checks poems database to see if given pair of words exists. If more than two words splits into all pairs
        Parameters
        ----------
        words (string or list) of words to check
        ret_lines (optional) - whether or not to return the lines in which the phrase occurs. Only used for testing
        include_syns (optional) - whether or not to also check for synonyms of phrase words
        """
        if type(words) == list:
            if len(words) == 1: return True
            words = " ".join(words)
        words_arr = words.split()
        if len(words_arr) > 2: return self.phrase_in_poem(words_arr[:2]) and self.phrase_in_poem(words_arr[1:])
        if words_arr[0] == words_arr[1]: return True
        if words_arr[0] in self.gender: return True #?
       # words = " "+ words + " "

        #print("evaluating", words)


        cases = []
        if include_syns:
            syns = []
            for j in range(len(words_arr)):
                syns.append([l.name() for s in wn.synsets(words_arr[j]) for l in s.lemmas() if l.name() in self.dict_meters])
            contenders = [words.split()[0] + " " + w for w in syns[1]]
            contenders += [w + " " + words.split()[1] for w in syns[0]]

        for poem in self.poems:
            poem = " " + poem.lower() + " " #in case its right at beginning or end since we check one before and one after occurence
            i = poem.find(words)
            if i != -1 and poem[i-1] not in string.ascii_lowercase and poem[i+len(words)] not in string.ascii_lowercase: #poem has phrase and characters before and after phrase arent letters (ie spaces or punctuation)
                if not ret_lines: return True
                else:
                    for line in poem.split("\n"):
                        line = " " + line.lower() + " "
                        i = line.find(words)
                        if i != -1 and line[i - 1] not in string.ascii_lowercase and line[i + len(words)] not in string.ascii_lowercase:
                            cases.append(line)
            if include_syns:
                indexes = [poem.find(ws) for ws in contenders]
                if any([indexes[i] != -1 and poem[indexes[i] - 1] not in string.ascii_lowercase and poem[indexes[i] + len(contenders[i])] not in string.ascii_lowercase for i in range(len(indexes))]):
                    if not ret_lines: return True
                    else:
                        correct = contenders[np.argmax(indexes)]
                        for line in poem.split("\n"):
                            line = " " + line.lower() + " "
                            i = line.find(correct)
                            if i != -1 and line[i - 1] not in string.ascii_lowercase and line[i + len(correct)] not in string.ascii_lowercase:
                                cases.append(line)




        if len(cases) == 0: return False
        return cases

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
        words[word_number] = self.lang_vocab[np.argmax(outputs[out_number].detach().numpy())]

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

    def phrase_in_poem_fast(self, words, include_syns=False):
        if type(words) == list:
            if len(words) == 1: return True
            words = " ".join(words)
        words = words.split()
        if len(words) > 2: return self.phrase_in_poem_fast(words[:2]) and self.phrase_in_poem_fast(words[1:])
        if words[0] == words[1]: return True
        if words[0] in self.gender: return True  # ?
        # words = " "+ words + " "

        print("evaluating", words)

        if include_syns:
            syns = []
            for j in words:
                syns.append([l.name() for s in wn.synsets(j) for l in s.lemmas() if l.name() in self.dict_meters])
            contenders = [words[0] + " " + w for w in syns[1]]
            contenders += [w + " " + words[1] for w in syns[0]]
            print(words, ": " , contenders)
            return any(self.phrase_in_poem_fast(c) for c in contenders)



        if words[0] in self.surrounding_words:
            return words[1] in self.surrounding_words[words[0]]
        elif words[1] in self.surrounding_words:
            return words[0] in self.surrounding_words[words[1]]
        else:
            self.surrounding_words[words[0]] = []
            self.surrounding_words[words[1]] = []
            translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
            for poem in self.poems:
                poem = " " + poem.lower() + " "
                for word in words:
                    a = poem.find(word)
                    if a != -1 and poem[a-1] not in string.ascii_lowercase and poem[a+len(word)] not in string.ascii_lowercase:
                        #print(a, poem[a:a+len(word)])
                        p_words = poem.translate(translator).split() #remove punctuation and split
                        if word in p_words: #not ideal but eg a line which ends with a '-' confuses it
                            a = p_words.index(word)
                            if a - 1 >= 0 and a - 1 < len(p_words): self.surrounding_words[word].append(p_words[a-1])
                            if a + 1 >= 1 and a + 1 < len(p_words): self.surrounding_words[word].append(p_words[a+1])
            return self.phrase_in_poem_fast(words, include_syns=False)

    def reset_number_words(self):
        for pos in list(self.pos_to_words):
            if "_" in pos: #or pos in "0123456789"
                del self.pos_to_words[pos]

    def my_softmax(self, x, exclude_zeros=False):
        """Compute softmax values for each sets of scores in x.
           exclude_zeros (bool) retains zero elements
        """
        if exclude_zeros and max(x) <= 0:
            print("max <=0 so retrying without exclusion")
            return softmax(x)  # has to be at least one non negative
            # elif ma == min(x): return np.array(x)/sum(x)
        else:
            e_x = np.exp(x - np.max(x))
            if exclude_zeros:
                e_x[np.array(x) == 0] = 0
        return e_x / e_x.sum()

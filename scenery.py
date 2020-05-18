import pickle
import random
import torch
import json
import numpy as np
import pandas as pd

from py_files import helper

from transformers import BertTokenizer, BertForMaskedLM
from transformers import RobertaTokenizer, RobertaForMaskedLM

import pronouncing

from nltk.corpus import wordnet as wn
from nltk import PorterStemmer


from pattern.en import comparative, superlative, pluralize



class Scenery_Gen():
    def __init__(self, model="bert", postag_file='saved_objects/postag_dict_all+VBN.p',
                 syllables_file='saved_objects/cmudict-0.7b.txt',
                 extra_stress_file='saved_objects/edwins_extra_stresses.txt',
                 top_file='saved_objects/words/top_words.txt',
                 templates_file='poems/scenery_templates.txt'):

        #self.templates = [("FROM scJJS scNNS PRP VBZ NN", "0_10_10_1_01_01"),
         #                 ("THAT scJJ scNN PRP VBD MIGHT RB VB", "0_10_10_1_0_10_1"),
          #                ("WHERE ALL THE scNNS OF PRP$ JJ NNS", "0_1_0_10_1_0_10_1"),
           #               ("AND THAT JJ WHICH RB VBZ NN", "0_1_01_0_10_1_01")]
        with open(templates_file) as tf:
            self.templates = [(" ".join(line.split()[:-1]), line.split()[-1]) for line in tf.readlines()]
        with open(postag_file, 'rb') as f:
            self.postag_dict = pickle.load(f)
        self.pos_to_words = self.postag_dict[1]
        self.words_to_pos = self.postag_dict[2]

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


        else:
            self.lang_model = None

        self.poems = pd.read_csv('poems/kaggle_poem_dataset.csv')['Content']

        with open(top_file) as tf:
            self.top_common_words = [line.strip() for line in tf.readlines()][:125]

        self.stemmer = PorterStemmer()

        self.api_url = 'https://api.datamuse.com/words'



    def get_word_pos(self, word):
        """
        Get the set of POS category of a word. If we are unable to get the category, return None.
        """
        # Special case
        if word.upper() in self.special_words:
            return [word.upper()]
        if word not in self.words_to_pos:
            return None
        return self.words_to_pos[word]

    def get_pos_words(self,pos, meter=None):
        """
        Gets all the words of a given POS
        Parameters
        ----------
        pos - the POS you want
        meter - (optional) returns only words which fit the given meter, e.g. 101
        """
        if pos in self.special_words:
            return [pos.lower()]
        if pos not in self.pos_to_words:
            return None
        if meter:
            ret = [word for word in self.pos_to_words[pos] if word in self.dict_meters and meter in self.dict_meters[word]]
            if len(ret) == 0:
                return False
            return ret
        return self.pos_to_words[pos]

    def getRhymes(self, theme, words):
        """
        :param theme: an array of either [prompt] or [prompt, line_theme] to find similar words to. JUST PROMPT FOR NOW
        :return: all words which rhyme with similar words to the theme in format {similar word: [rhyming words], similar word: [rhyming words], etc.}
        """
        if len(theme) > 1:
            prompt = theme[0]
            tone = theme[1:]
        else:
            prompt = theme[0]
            tone = "NONE"
        try:
            with open("saved_objects/saved_rhymes", "rb") as pickle_in:
                mydict = pickle.load(pickle_in)

        except:
            with open("saved_objects/saved_rhymes", "wb") as pickle_in:
                mydict = {}
                pickle.dump(mydict, pickle_in)
        if prompt not in mydict.keys():
            mydict[prompt] = {}
        if tone not in mydict[prompt].keys():
            print("havent stored anything for ", theme, "please wait...")
            print(" (ignore the warnings) ")
            words = helper.get_similar_word_henry(theme, n_return=30, word_set=set(words))
            w_rhyme_dict = {w3: {word for word in helper.get_rhyming_words_one_step_henry(self.api_url, w3) if
                                   word in self.words_to_pos and word in self.dict_meters and word not in self.top_common_words[:70]} for #deleted: and self.filter_common_word_henry(word, fast=True)
                              w3 in words if w3 not in self.top_common_words[:70] and w3 in self.dict_meters}

            #if len(w_rhyme_dict) > 0:
            mydict[prompt][tone] = {k: v for k, v in w_rhyme_dict.items() if len(v) > 0}

        with open("saved_objects/saved_rhymes", "wb") as pickle_in:
            pickle.dump(mydict, pickle_in)
        return mydict[prompt][tone]

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
                while not self.suitable_last_word(last_word_dict[i][0], i-1) or last_word_dict[i][0] in first_rhymes:
                        #or any(rhyme_dict['A'][last_word_dict[i][0]] in rhyme_dict['A'][word] for word in first_rhymes):
                    last_word_dict[i] = [random.choice(list(rhyme_dict[scheme[i]].keys()))]
                    if not any(self.templates[i-1][1].split("_")[-1] in self.dict_meters[w] for w in rhyme_dict[scheme[i]]):
                        word = last_word_dict[i][0]
                        if self.templates[i-1][0].split()[-1] in self.get_word_pos(word) and len(self.dict_meters[word][0]) == len(self.templates[i-1][1].split("_")[-1]) and any(self.suitable_last_word(r, i+1) for r in rhyme_dict[scheme[i]][word]):
                            self.dict_meters[word].append(self.templates[i-1][1].split("_")[-1])
                            print("cheated with ", word, " ", self.dict_meters[word], self.suitable_last_word(word, i-1))
                    j +=1
                    if j > len(rhyme_dict[scheme[i]]) * 2: input(str(scheme[i]) + " " + str(rhyme_dict[scheme[i]]))
                first_rhymes.append(last_word_dict[i][0])

            if i in [3, 4]:  # lines with an old rhyme -> pick a random value corresponding to key of rhyming couplet
                letter = scheme[i]
                pair = last_word_dict[i - 2][0]
                last_word_dict[i] = [word for word in rhyme_dict[letter][pair] if self.suitable_last_word(word, i-1)]
                if len(last_word_dict[i]) == 0:
                    print("fuck me", last_word_dict, i, self.templates[i])
                    print(1/0)
        return last_word_dict

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
        rhymes = []
        for line_number, (template, meter) in enumerate(self.templates):
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
                        print(thematic)
                        if theme_words[thematic] > 1 and meter[i] in self.dict_meters[thematic] and pos in self.get_word_pos(thematic) :
                            new_words.append(thematic)
                            scores.append(theme_words[thematic])
                            #if verbose: print("found ", thematic, theme_words[thematic], "for ", meter[i], template[i])
               #print(new_words, meter[i], pos)
                if len(new_words) == 0: new_word = random.choice(self.get_pos_words(pos, meter=meter[i]))
                else:
                    dist = helper.softmax(scores)
                    new_word = np.random.choice(new_words, p=dist)
                    theme_words[new_word] = 0 #don't choose same word twice
                    theme_words[self.stemmer.stem(new_word)] = 0
                    theme_words[pluralize(new_word)] = 0
                    print(new_word, " chosen with prob", dist[new_words.index(new_word)])
                line += new_word + " "

            if line_number % 4 < 2:
                word = None
                #high_score = 0
                rhyme_pos = self.templates[min(13, line_number + 2)][0].split()[-1]
                rhyme_met = self.templates[min(13, line_number + 2)][1].split("_")[-1]
                while not word or not any(r in self.get_pos_words(rhyme_pos, meter=rhyme_met) for r in pronouncing.rhymes(word)):
                    word = random.choice(self.get_pos_words(template[-1], meter=meter[-1]))
                line += word
                rhymes.append(word)
            else:
                n = -2
                if line_number == 13: n = -1
                line += random.choice([rhyme for rhyme in self.get_pos_words(template[-1], meter=meter[-1]) if self.rhymes(rhyme, lines[n].split()[-1])])
            #line += random.choice(last_word_dict[line_number])
            print("line initially ", line)
            rhyme_set = []
            for r in rhymes:
                rhyme_set += pronouncing.rhymes(r)
            print("rhyme_set length: ", len(rhyme_set))
            if self.lang_model: line = self.update_bert(line.strip().split(), meter, template, len(template), theme_words=theme_words, rhyme_words=rhyme_set, filter_meter=True, verbose=True)
            print("line after ", line)
            lines.append(line)
            #break
        print(lines)

    def update_bert(self, line, meter, template, iterations, theme_words=[], rhyme_words = [], filter_meter=True, verbose=False):
        if iterations == 0: return " ".join(line) #base case
        #TODO deal with tags like ### (which are responsible for actually cool words)
        input_ids = torch.tensor(self.tokenizer.encode(line, add_special_tokens=False)).unsqueeze(0) #tokenizes
        loss, outputs = self.lang_model(input_ids, masked_lm_labels=input_ids) #masks each token and gives probability for all tokens in each word. Shape num_words * vocab_size
        if verbose: print("loss = ", loss)
        softmax = torch.nn.Softmax(dim=1) #normalizes the probabilites to be between 0 and 1
        outputs = softmax(outputs[0])
        #for word_number in range(0,len(line)-1): #ignore  last word to keep rhyme
        #word_number = random.choice(np.arange(len(line)-1))
        word_number = np.argmin(np.array([outputs[i][self.vocab_to_num[line[i]]] for i in range(len(line)) ]))
        temp = template[word_number].split("sc")[-1]
        if len(self.get_pos_words(temp)) > 1 and temp not in ['PRP', 'PRP$']: #only change one word each time?
            filt = np.array([int(word in self.words_to_pos and temp in self.get_word_pos(word)) for word in self.lang_vocab])
            if filter_meter: filt *= np.array([int(word in self.dict_meters and meter[word_number] in self.dict_meters[word]) for word in self.lang_vocab])
            predictions = outputs[word_number].detach().numpy() * filt #filters non-words and words which dont fit meter and template
            #TODO add theme relevance weighting. add internal rhyme and poetic device weighting
            for p in range(len(predictions)):
                if predictions[p] > 0.001 and self.lang_vocab[p] in rhyme_words:
                    if verbose: print("weighting internal rhyme '", self.lang_vocab[p], "', orig: ", predictions[p])
                    predictions[p] *= 2
                if predictions[p] > 0.001 and self.lang_vocab[p] in theme_words and "sc" in template[word_number]:
                    predictions[p] *= (1 + theme_words[self.lang_vocab[p]])
                    if verbose: print("weighting thematic '", self.lang_vocab[p], "', now: ", predictions[p])

            predictions /= sum(predictions)
            if verbose: print("min: ", min(predictions), " max: ", max(predictions), "sum: ", sum(predictions), ", ", sorted([self.lang_vocab[p] for p in range(len(predictions)) if predictions[p] > 0], reverse=True))
            if iterations > 1:
                line[word_number] = np.random.choice(self.lang_vocab, p=predictions)
            else: #greedy for last iteration
                line[word_number] = self.lang_vocab[np.argmax(predictions)]
            if verbose: print("word now ", line[word_number], "prob: ", predictions[self.lang_vocab.index(line[word_number])])

            if verbose: print("line now", line)
        else:
            iterations += 1
        return self.update_bert(line, meter, template, iterations-1, theme_words=theme_words, rhyme_words=rhyme_words, verbose=verbose)

    def rhymes(self, word1, word2):
        return word1 in pronouncing.rhymes(word2) or word2 in pronouncing.rhymes(word1)

    def get_theme_words(self, theme, k=1, verbose=True, theme_file="saved_objects/theme_words.p", extras_file='saved_objects/extra_adjs.p'):
        try:
            with open(theme_file, "rb") as pickle_in:
                print("loading from file")
                theme_word_dict = pickle.load(pickle_in)

        except:
            with open(theme_file, "wb") as pickle_in:
                theme_word_dict = {}
                pickle.dump(theme_word_dict, pickle_in)
        try:
            with open(extras_file, "rb") as p_in:
                extras = pickle.load(p_in)
        except:
            with open(extras_file, "wb") as p_in:
                extras = {}
                pickle.dump(extras, p_in)
        if theme not in theme_word_dict:
            print(theme, "not in file. Generating...")

            syn = wn.synsets(theme)
            theme_syns = [l.name() for s in syn for l in s.lemmas() if l.name() in self.dict_meters]
            cases = []
            for poem in self.poems: #find poems which have theme syns
                if any(word in poem for word in theme_syns):
                    for line in poem.split("\n"): #find lines which have theme syns
                        if any(word in line for word in theme_syns):
                            cases.append(line)
            print("theme_syns" , theme_syns)
            print(cases)
            theme_words = {}
            for case in cases:
                words = case.split()
                for i in range(len(words)):
                    if words[i] in theme_syns:
                        good_pos = ['JJ', 'JJS', 'RB', 'VB', 'VBP', 'VBD', 'VBZ', 'VBG', 'NN', 'NNS']
                        punct = [".", ",", "?", "-", "!"]
                        new_words = [words[i]]
                        left = i - 1
                        while left >= max(0, i-k):
                            if words[left] in punct: left = max(0, left-1)
                            if words[left] in self.words_to_pos and words[left] in self.dict_meters and words[left] not in self.top_common_words and any(pos in good_pos for pos in self.get_word_pos(words[left])):
                                new_words.append(words[left])
                            left -=1
                        right = i + 1
                        while right <= min(len(words) -1, i+k):
                            if words[right] in punct: right = max(len(words) - 1, right + 1)
                            if words[right] in self.words_to_pos and words[right] in self.dict_meters and words[right] not in self.top_common_words and any(pos in good_pos for pos in self.get_word_pos(words[right])):
                                new_words.append(words[right])
                            right += 1
                        for w in new_words:
                            if not self.get_word_pos(w) or w not in self.dict_meters: continue
                            if w not in theme_words: theme_words[w] = 0
                            theme_words[w] = min(theme_words[w] + 1, 20)
                            if "JJ" in self.get_word_pos(w):
                                new_words.append(comparative(w))
                                #self.words_to_pos[comparative(w)] = ["JJR"]
                                #self.pos_to_words["JJR"].append(comparative(w))
                                extras[comparative(w)] = ["JJR"]

                                new_words.append(superlative(w))
                                #self.words_to_pos[superlative(w)] = ["JJS"]
                                #self.pos_to_words["JJS"].append(superlative(w))
                                extras[superlative(w)] = ["JJS"]

                                #print("adding ", new_words[-2:])
                            elif "NN" in self.get_word_pos(w):
                                if pluralize(w) != w:
                                    new_words.append(pluralize(w))
                                    #print("adding ", new_words[-1])
                            else:
                                st = self.stemmer.stem(w)
                                if st not in new_words:
                                    new_words.append(st)
                                    #print("adding ", new_words[-1])


            #keep only the ones that come up as synonyms for at least two?
            theme_words["purple"] = 0 # comes up weirdly often
            theme_word_dict[theme] = theme_words
            for w in theme_word_dict[theme]:
                theme_word_dict[theme][w] *= abs(helper.get_spacy_similarity(theme, w))
            with open(extras_file, 'wb') as f:
                pickle.dump(extras, f)
        with open(theme_file, "wb") as pickle_in:
            pickle.dump(theme_word_dict, pickle_in)

        for extra in extras:
            self.words_to_pos[extra] = extras[extra]
            self.pos_to_words[extras[extra][0]].append(extra)
        return theme_word_dict[theme]




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
    def __init__(self, model=None, postag_file='saved_objects/postag_dict_all+VBN.p',
                 syllables_file='saved_objects/cmudict-0.7b.txt',
                 extra_stress_file='saved_objects/edwins_extra_stresses.txt',
                 top_file='saved_objects/words/top_words.txt'):
        #self.templates = [("FROM JJS NNS PRP VBP NN", "0_10_10_1_01_01"),
         #                 ("THAT RB NN POS VBD MIGHT RB VB", "0_10_10__1_0_10_1"),
          #                ("WHERE ALL THE NN OF PRP$ JJ NNS", "0_1_0_10_1_0_10_1"),
           #               ("AND THAT JJ WHICH RB VBZ VB", "0_1_01_0_10_1_01")]
        self.templates = [("FROM scJJS scNNS PRP VBZ NN", "0_10_10_1_01_01"),
                          ("THAT scJJ scNN PRP VBD MIGHT RB VB", "0_10_10_1_0_10_1"),
                          ("WHERE ALL THE scNNS OF PRP$ JJ NNS", "0_1_0_10_1_0_10_1"),
                          ("AND THAT JJ WHICH RB VBZ NN", "0_1_01_0_10_1_01")]
        with open(postag_file, 'rb') as f:
            postag_dict = pickle.load(f)
        self.pos_to_words = postag_dict[1]
        self.words_to_pos = postag_dict[2]

        self.special_words = helper.get_finer_pos_words()

        self.dict_meters = helper.create_syll_dict([syllables_file], extra_stress_file)

        if model == "bert":
            self.lang_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.lang_vocab = list(self.tokenizer.vocab.keys())
            self.lang_model.eval()

        elif model == "roberta":
            self.lang_model = RobertaForMaskedLM.from_pretrained('roberta-large') # 'roberta-base'
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
            with open("saved_objects/roberta/vocab.json") as json_file:
                j = json.load(json_file)
            self.lang_vocab = list(j.keys())
            self.lang_model.eval()




        self.poems = pd.read_csv('poems/kaggle_poem_dataset.csv')['Content']

        with open(top_file) as tf:
            self.top_common_words = [line.strip() for line in tf.readlines()][:125]

        self.stemmer = PorterStemmer()




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
        theme_words = self.get_theme_words(theme, verbose=False)
        print(theme_words)
        lines = []
        rhymes = []
        for template, meter in self.templates:
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
            if len(lines) < 2:
                word = None
                #high_score = 0
                rhyme_pos = self.templates[len(lines) + 2][0].split()[-1]
                rhyme_met = self.templates[len(lines) + 2][1].split("_")[-1]
                """for thematic in theme_words:
                    if theme_words[thematic] > high_score and meter[-1] in self.dict_meters[thematic] and template[-1] in self.get_word_pos(thematic):
                        if any(r in self.get_pos_words(rhyme_pos, meter=rhyme_met) for r in pronouncing.rhymes(thematic)):
                            word = thematic
                            high_score = theme_words[thematic]"""

                while not word or not any(r in self.get_pos_words(rhyme_pos, meter=rhyme_met) for r in pronouncing.rhymes(word)):
                    word = random.choice(self.get_pos_words(template[-1], meter=meter[-1]))
                line += word
                rhymes.append(word)
            else:
                line += random.choice([rhyme for rhyme in self.get_pos_words(template[-1], meter=meter[-1]) if self.rhymes(rhyme, lines[-2].split()[-1])])
            #print("line initially ", line)
            rhyme_set = []
            for r in rhymes:
                rhyme_set += pronouncing.rhymes(r)
            #print("rhyme_set length: ", len(rhyme_set))
            #line = self.update_bert(line.strip().split(), meter, template, len(template), rhyme_words=rhyme_set, filter_meter=True, verbose=True)
            #print("line after ", line)
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
        word_number = random.choice(np.arange(len(line)-1))
        if len(self.get_pos_words(template[word_number])) > 1 and template[word_number] not in ['PRP', 'PRP$']: #only change one word each time?
            filt = np.array([int(word in self.words_to_pos and template[word_number] in self.get_word_pos(word)) for word in self.lang_vocab])
            if filter_meter: filt *= np.array([int(word in self.dict_meters and meter[word_number] in self.dict_meters[word]) for word in self.lang_vocab])
            predictions = outputs[word_number].detach().numpy() * filt #filters non-words and words which dont fit meter and template
            #TODO add theme relevance weighting. add internal rhyme and poetic device weighting
            for p in range(len(predictions)):
                if predictions[p] > 0.001 and self.lang_vocab[p] in rhyme_words:
                    if verbose: print("weighting internal rhyme '", self.lang_vocab[p], "', orig: ", predictions[p])
                    predictions[p] *= 2
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

    def get_theme_words(self, theme, k=1, verbose=True, theme_file="saved_objects/theme_words.p"):
        try:
            with open(theme_file, "rb") as pickle_in:
                print("loading from file")
                theme_word_dict = pickle.load(pickle_in)

        except:
            with open(theme_file, "wb") as pickle_in:
                theme_word_dict = {}
                pickle.dump(theme_word_dict, pickle_in)
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
                                self.words_to_pos[comparative(w)] = ["JJR"]
                                new_words.append(superlative(w))
                                self.words_to_pos[superlative(w)] = ["JJS"]
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
        with open(theme_file, "wb") as pickle_in:
            pickle.dump(theme_word_dict, pickle_in)

        return theme_word_dict[theme]



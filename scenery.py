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

import pronouncing

from nltk.corpus import wordnet as wn
from nltk import PorterStemmer


class Scenery_Gen():
    def __init__(self, model=None, postag_file='saved_objects/postag_dict_all+VBN.p',
                 syllables_file='saved_objects/cmudict-0.7b.txt',
                 extra_stress_file='saved_objects/edwins_extra_stresses.txt',
                 top_file='saved_objects/words/top_words.txt',
                 #templates_file='poems/scenery_templates.txt',
                 templates_file='poems/number_templates.txt',
                 mistakes_file='saved_objects/mistakes.txt'):

        #self.templates = [("FROM scJJS scNNS PRP VBZ NN", "0_10_10_1_01_01"),
         #                 ("THAT scJJ scNN PRP VBD MIGHT RB VB", "0_10_10_1_0_10_1"),
          #                ("WHERE ALL THE scNNS OF PRP$ JJ NNS", "0_1_0_10_1_0_10_1"),
           #               ("AND THAT JJ WHICH RB VBZ NN", "0_1_01_0_10_1_01")]
        with open(templates_file) as tf:
            #lines = tf.read()
                #self.templates = {dev.split()[0]: [(" ".join(t.split()[:-1]), t.split()[-1] for t in dev[1:].split("\n"))] for dev in lines.split("#") if len(dev) > 0}
            self.templates = [(" ".join(line.split()[:-1]), line.split()[-1]) for line in tf.readlines() if "#" not in line and len(line) > 1]
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


        else:
            self.lang_model = None

        #with open('poems/kaggle_poem_dataset.csv', newline='') as csvfile:
         #   self.poems = csv.DictReader(csvfile)
        self.poems = list(pd.read_csv('poems/kaggle_poem_dataset.csv')['Content'])
        self.surrounding_words = {}

        with open(top_file) as tf:
            self.top_common_words = [line.strip() for line in tf.readlines()][:125]

        self.stemmer = PorterStemmer()

        self.api_url = 'https://api.datamuse.com/words'

        self.gender = random.choice([["he", "him", "his", "himself"], ["she", "her", "hers", "herself"]])



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

    def get_pos_words(self,pos, meter=None, phrase=()):
        """
        Gets all the words of a given POS
        Parameters
        ----------
        pos - the POS you want
        meter - (optional) returns only words which fit the given meter, e.g. 101
        phrase (optional) - returns only words which have a phrase in the dataset. in format ([word1, word2, word3], i) where i is the index of the word to change since the length can be 2 or 3
        """
        #print("oi," , pos, meter, phrase)
        if pos in self.special_words:
            return [pos.lower()]
        if "PRP" in pos:
            ret = [p for p in self.pos_to_words[pos] if meter and p in self.gender and meter in self.get_meter(p) ]
            if len(ret) == 0: ret = [input("PRP not happening " + pos + " '" + meter + "' " + str(self.gender) + str([self.dict_meters[p] for p in self.gender]))]
            return ret
        if pos not in self.pos_to_words:
            return None
        if meter:
            ret = [word for word in self.pos_to_words[pos] if word in self.dict_meters and meter in self.dict_meters[word]]
            if len(phrase) > 1 and len(phrase[0]) > 1:
                phrases = []
                for word in ret:
                    phrase[0][phrase[1]] = word
                    phrases.append(" ".join(phrase[0]))
                print(phrases, ret)
                ret = [ret[i] for i in range(len(ret)) if self.phrase_in_poem_fast(phrases[i], include_syns=True)]
            if len(ret) == 0:
                return None
            return ret
        return self.pos_to_words[pos]

    def get_meter(self, word):
        if word not in self.dict_meters: return []
        return self.dict_meters[word]

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

    def write_stanza(self, theme="flower", verbose=True, checks=["RB", "NNS"]):
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
        print("a")
        theme_gen = theme_word_file.Theme()
        theme_words = theme_gen.get_theme_words(theme, verbose=False)
        print(theme_words)
        lines = []
        orig_lines = []
        for line_number, (template, meter) in enumerate(self.templates):
            print("b")
            if line_number % 4 == 0: rhymes = []
            template = template.split()
            meter = meter.split("_")
            print("c", line_number)
            line = self.write_line(line_number, template, meter, rhymes=rhymes, theme_words=theme_words)
            print("d", line_number)
            if line_number % 4 < 2:
                rhymes.append(line.split()[-1])
            #line += random.choice(last_word_dict[line_number])
            #checks = ["RB", "NNS"]
            for check in checks:
                if check in template:
                    print("e")
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
                            input("not happening " + line + str(template) + str(meter) + meter[adv])
                            line = self.write_line(line_number, template, meter, rhymes=rhymes, theme_words=theme_words)
                            inc_syn = False
                            continue
                        line_arr[adv] = random.choice(poss)
                        print(check, " updated, line now ", " ".join(line_arr))
                    line = " ".join(line_arr)
            if self.lang_model:
                print("f")
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
        if self.lang_model: [print(orig_lines[i]) for i in range(len(orig_lines))]
        print("")
        print("         ---", theme.upper(), "---       ")
        for cand in range(len(lines)):
            print(lines[cand])
            if ((cand + 1) % 4 == 0): print("")

    def write_line(self, n, template, meter, rhymes=[], theme_words=[], numbers = {}, verbose=True):
        numbers = {}
        line = ""
        for i in range(len(template) - 1):
            new_words = []
            scores = []
            pos = template[i]
            num = -1
            letter = ""
            if "_" in pos:
                try: num = int(pos.split("_")[1])
                except: letter = pos.split("_")[1]
                pos = pos.split("_")[0]
            if "sc" in pos:
                pos = pos.split("sc")[-1]
                for thematic in theme_words[pos]:
                    if theme_words[pos][thematic] > 0.1 and meter[i] in self.dict_meters[thematic]:
                        new_words.append(thematic)
                        if num in numbers: #if numbered word exists
                            scores.append(helper.get_spacy_similarity(numbers[num], thematic))
                        else:
                            scores.append(theme_words[pos][thematic])
            if len(new_words) == 0:
                #new_word = random.choice(self.get_pos_words(pos, meter=meter[i]))
                poss = self.get_pos_words(pos, meter=meter[i])
                if not poss: input("help " + pos + meter[i])
                if num in numbers:
                    print("checking", numbers[num], ":", poss)
                    dist = [helper.get_spacy_similarity(numbers[num], w) for w in poss]
                    new_word = np.random.choice(poss, p=helper.softmax(dist))
                    if verbose: print("weighted choice of ", new_word, ", related to ", numbers[num], "with prob ", dist[poss.index(new_word)])
                elif letter in numbers:
                    new_word = numbers[letter]
                else:
                    new_word = random.choice(poss)
            else:
                dist = helper.softmax(scores)
                new_word = np.random.choice(new_words, p=dist)
                theme_words[pos][new_word] = theme_words[pos][new_word] / 4.0  # don't choose same word twice
                #theme_words[self.stemmer.stem(new_word)] = 0
                #if "NN" in self.get_word_pos(new_word): theme_words[pluralize(new_word)] = 0
                if verbose: print(new_word, " chosen with prob", dist[new_words.index(new_word)], "now: ", theme_words[pos][new_word])
            line += new_word + " "
            if num > -1 and num not in numbers:
                numbers[num] = str(new_word)
                if verbose: print("numbers now ", numbers)
            elif letter != "" and letter not in numbers:
                numbers[letter] = new_word
                if verbose: print("numbers now ", numbers)
        if n == -1:
            word = None
            pos = template[-1].split("sc")[-1]
            num = -1
            letter = ""
            if "_" in pos:
                try: num = int(pos.split("_")[1])
                except: letter = pos.split("_")[1]
                pos = pos.split("_")[0]
            if num in numbers:
                word = numbers[num]
            elif letter in numbers:
                word = numbers[letter]
            else:
                word = random.choice(self.get_pos_words(pos, meter=meter[-1]))

        elif n % 4 < 2:
            word = None
            # high_score = 0
            pos = template[-1].split("sc")[-1]
            num = -1
            letter = ""
            if "_" in pos:
                try: num = int(pos.split("_")[1])
                except: letter = pos.split("_")[1]
                pos = pos.split("_")[0]

            rhyme_pos = self.templates[min(13, n + 2)][0].split()[-1].split("sc")[-1]
            rhyme_met = self.templates[min(13, n + 2)][1].split("_")[-1]
            while not word or not any(r in self.get_pos_words(rhyme_pos, meter=rhyme_met) for r in pronouncing.rhymes(word)):
                #word = random.choice(self.get_pos_words(pos, meter=meter[-1]))
                poss = self.get_pos_words(pos, meter=meter[-1])
                if num in numbers:
                    dist = [helper.get_spacy_similarity(numbers[num], w) for w in poss]
                    word = np.random.choice(poss, p=helper.softmax(dist))
                    if verbose: print("rhyme - weighted choice of ", word, ", related to ", numbers[num], "with prob ", dist[poss.index(word)])
                elif letter in numbers:
                    word = numbers[letter]
                else:
                    word = random.choice(poss) #could get stuck here !!
        else:
            r = (n % 4) - 2
            if n == 13: r = 0
            word = random.choice([rhyme for rhyme in self.get_pos_words(template[-1].split("sc")[-1], meter=meter[-1]) if self.rhymes(rhyme, rhymes[r])])
        line += word
        return line

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
                    input("change here and for the print")
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

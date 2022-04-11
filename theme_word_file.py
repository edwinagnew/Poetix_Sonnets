import pickle
import pandas as pd
import random
import string

import helper

from pattern.en import comparative, superlative, pluralize
from nltk.corpus import wordnet as wn
from nltk import PorterStemmer

import poem_core

class Theme(poem_core.Poem):

    def __init__(self, theme=None, postag_file='saved_objects/postag_dict_all+VBN.p',
                 mistakes_file='saved_objects/mistakes.txt',
                 syllables_file='saved_objects/cmudict-0.7b.txt',
                 extra_stress_file='saved_objects/edwins_extra_stresses.txt',
                 top_file='saved_objects/words/top_words.txt', ):

        poem_core.Poem.__init__(self, words_file="saved_objects/tagged_words.p",
                                syllables_file=syllables_file, extra_stress_file=extra_stress_file, top_file=top_file)



        self.poems = list(pd.read_csv('poems/kaggle_poem_dataset.csv')['Content'])

        self.stemmer = PorterStemmer()

        self.most_recent_theme_words = None
        if theme:
            self.most_recent_theme_words = self.get_theme_words(theme)

        #self.score_func = MethodType(score_function, self)



    def get_theme_words(self, theme, k=1, verbose=False, max_val=20, theme_file="saved_objects/theme_words.p", extras_file='saved_objects/extra_adjs.p'):
        if not theme or len(theme) == 0: return {}
        try:
            with open(theme_file, "rb") as pickle_in:
                if verbose: print("loading words from file")
                theme_word_dict = pickle.load(pickle_in)
            with open(extras_file, "rb") as p_in:
                extras = pickle.load(p_in)

        except:
            if verbose: print("either file not found")
            with open(theme_file, "wb") as pickle_in:
                theme_word_dict = {}
                pickle.dump(theme_word_dict, pickle_in)
            with open(extras_file, "wb") as p_in:
                extras = {}
                pickle.dump(extras, p_in)

        if theme not in theme_word_dict:
            if verbose: print(theme, "not in file. Generating...")


            cases = self.get_cases(theme)
            syn = wn.synsets(theme)
            theme_syns = [l.name() for s in syn for l in s.lemmas() if l.name() in self.dict_meters]

            if verbose:
                print("theme_syns" , theme_syns)
                print(cases)
            theme_words = {}
            for case in cases:
                words = case.split()
                for i in range(len(words)):
                    if words[i] in theme_syns:
                        good_pos = ['JJ', 'JJR', 'JJS', 'RB', 'VB', 'VBP', 'VBD', 'VBZ', 'VBG', 'NN', 'NNS', 'ABNN']
                        punct = string.punctuation
                        new_words = [words[i]]
                        left = i - 1
                        while left >= max(0, i-k):
                            if words[left] in punct: left = max(0, left-1)
                            if words[left] in self.words_to_pos and words[left] in self.dict_meters and words[left] not in self.top_common_words and any(pos in good_pos for pos in self.get_word_pos(words[left])):
                                new_words.append(words[left])
                            left -=1
                        right = i + 1
                        while right <= min(len(words) -1, i+k):
                            if words[right] in punct: right = min(len(words) - 1, right + 1)
                            if words[right] in self.words_to_pos and words[right] in self.dict_meters and words[right] not in self.top_common_words and any(pos in good_pos for pos in self.get_word_pos(words[right])):
                                new_words.append(words[right])
                            right += 1
                        for w in new_words:
                            if not self.get_word_pos(w) or w not in self.dict_meters: continue
                            p = self.get_word_pos(w)
                            if len(p) == 0:
                                print("you're fucked with ", w, p)
                                print(1/0)
                            p = random.choice(p)
                            if p not in theme_words:
                                theme_words[p] = {}
                            if w not in theme_words[p]: theme_words[p][w] = 0
                            theme_words[p][w] = min(theme_words[p][w] + 1, max_val)
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
                                if pluralize(w) != w and w[-1] != 's':
                                    new_words.append(pluralize(w))
                                    extras[pluralize(w)] = ["NNS"]
                                    #print("adding ", new_words[-1])
                            else:
                                st = self.stemmer.stem(w)
                                if st not in new_words:
                                    new_words.append(st)
                                    #print("adding ", new_words[-1])


            #keep only the ones that come up as synonyms for at least two?
            theme_word_dict[theme] = theme_words
            if verbose: print("got themes", theme_word_dict)
            for p in theme_word_dict[theme]:
                for w in theme_word_dict[theme][p]:
                    theme_word_dict[theme][p][w] *= abs(helper.get_spacy_similarity(theme, w))#/max_val
            with open(extras_file, 'wb') as f:
                pickle.dump(extras, f)
        with open(theme_file, "wb") as pickle_in:
            pickle.dump(theme_word_dict, pickle_in)

        for extra in extras:
            self.words_to_pos[extra] = extras[extra]
            self.pos_to_words[extras[extra][0]][extra] = 1
        return theme_word_dict[theme]


    def score_theme_word(self, word, theme):
        if not self.most_recent_theme_words or theme not in self.most_recent_theme_words:
            self.most_recent_theme_words = self.get_theme_words(theme)
        for p in self.most_recent_theme_words[theme]:
            if word in self.most_recent_theme_words[theme][p]: return self.most_recent_theme_words[theme][p][word]
        return 0

    def get_cases(self, theme):
        if " " in theme:
            ret = []
            for t in theme.split(): ret += self.get_cases(t)
            return ret
        syn = wn.synsets(theme)
        theme_syns = [l.name() for s in syn for l in s.lemmas() if l.name() in self.dict_meters]
        cases = []
        for poem in self.poems:  # find poems which have theme syns
            if any(word in poem for word in theme_syns):
                for line in poem.split("\n"):  # find lines which have theme syns
                    if any(word in line for word in theme_syns):
                        cases.append(line)
        return cases

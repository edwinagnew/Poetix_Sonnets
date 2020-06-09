from py_files import helper

import random
import pickle
import numpy as np

class Poem:
    def __init__(self, words_file="saved_objects/tagged_words.p",
                 templates_file='poems/number_templates.txt',
                 syllables_file='saved_objects/cmudict-0.7b.txt',
                 extra_stress_file='saved_objects/edwins_extra_stresses.txt',
                 top_file='saved_objects/words/top_words.txt'):
        self.pos_to_words, self.words_to_pos = helper.get_new_pos_dict(words_file)

        with open(templates_file) as tf:
            self.templates = [(" ".join(line.split()[:-1]), line.split()[-1]) for line in tf.readlines() if "#" not in line and len(line) > 1]

        self.special_words = helper.get_finer_pos_words()

        self.dict_meters = helper.create_syll_dict([syllables_file], extra_stress_file)

        self.gender = random.choice([["he", "him", "his", "himself"], ["she", "her", "hers", "herself"]])

        with open(top_file) as tf:
            self.top_common_words = [line.strip() for line in tf.readlines()][:125]


        self.api_url = 'https://api.datamuse.com/words'

        with open("poems/end_pos.txt", "r") as pickin:
            lis = pickin.readlines()
            self.end_pos = {}
            for l in lis:
                self.end_pos[l.split()[0]] = l.split()[1:]

    def get_meter(self, word):
        if word not in self.dict_meters: return []
        return self.dict_meters[word]

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

    def get_pos_words(self,pos, meter=None):
        """
        Gets all the words of a given POS
        Parameters
        ----------
        pos - the POS you want
        meter - (optional) returns only words which fit the given meter, e.g. 101
        """
        #print("oi," , pos, meter, phrase)
        if pos in self.special_words:
            return [pos.lower()]
        if "PRP" in pos:
            ret = [p for p in self.pos_to_words[pos] if meter and p in self.gender and meter in self.get_meter(p) ]
            if len(ret) == 0: ret = [input("PRP not happening " + pos + " '" + meter + "' " + str(self.gender) + str([self.dict_meters[p] for p in self.gender]))]
            return ret
        if pos not in self.pos_to_words:
            return []
        if meter:
            ret = [word for word in self.pos_to_words[pos] if word in self.dict_meters and meter in self.dict_meters[word]]
            if len(ret) == 0:
                return []
            return ret
        return [p for p in self.pos_to_words[pos]]

    def weighted_choice(self,pos, meter=None):
        poss = self.get_pos_words(pos, meter=meter)
        if not poss: return None
        poss_dict = {p:self.pos_to_words[pos][p] for p in poss}
        vals = poss_dict.values()
        if min(vals) == max(vals): return random.choice(vals)
        else:
            return np.random.choice(poss_dict.keys(), p=helper.softmax(vals))

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

    def last_word_dict(self, rhyme_dict, scheme = {1: 'A', 2: 'B', 3: 'A', 4: 'B', 5: 'C', 6: 'D', 7: 'C', 8: 'D', 9: 'E', 10: 'F', 11: 'E', 12: 'F', 13: 'G', 14: 'G'}):
        """
        Given the rhyme sets, extract all possible last words from the rhyme set
        dictionaries.

        Parameters
        ----------
        scheme rhyme_scheme
        rhyme_dict: dictionary
            Format is   {'A': {tone1 : {similar word: [rhyming words], similar word: [rhyming words], etc.}}, {tone2:{...}}},
                        'B': {tone1 : {similar word: [rhyming words], similar word: [rhyming words], etc.}}, {tone2:{...}}}
                        etc
        Returns
        -------
        dictionary
            Format is {1: ['apple', 'orange'], 2: ['apple', orange] ... }

        """
        last_word_dict={}

        print(rhyme_dict)
        first_rhymes = []
        for i in range(1,15):
            if i in [1, 2, 5, 6, 9, 10, 13]:  # lines with a new rhyme -> pick a random key
                last_word_dict[i] = [random.choice(list(rhyme_dict[scheme[i]].keys()))] #NB ensure it doesnt pick the same as another one
                while not self.suitable_last_word(last_word_dict[i][0]) or last_word_dict[i][0] in first_rhymes or any(rhyme_dict['A'][last_word_dict[i][0]] in rhyme_dict['A'][word] for word in first_rhymes):
                    last_word_dict[i] = [random.choice(list(rhyme_dict[scheme[i]].keys()))]
                first_rhymes.append(last_word_dict[i][0])
            if i in [3, 4, 7, 8, 11, 12, 14]:  # lines with an old rhyme -> pick a random value corresponding to key of rhyming couplet
                letter = scheme[i]
                pair = last_word_dict[i-2][0]
                if i == 14:
                    pair = last_word_dict[13][0]
                print(i, letter, pair, rhyme_dict[letter][pair])
                last_word_dict[i] = [word for word in rhyme_dict[letter][pair] if self.suitable_last_word(word)]
        return last_word_dict

    def suitable_last_word(self, word): #checks pos is in self.end_pos and has correct possible meters
        return any(w in self.end_pos.keys() for w in self.get_word_pos(word)) and any(t in self.end_pos[pos] for t in self.dict_meters[word] for pos in self.get_word_pos(word) if pos in self.end_pos)



import numpy as np
import pickle
from collections import defaultdict
from py_files import helper
import random
import character_gen
from py_files import line
#from sonnet_basic import *
#from sonnet_basic import *

class Sonnet_Gen():
    def __init__(self,postag_file='saved_objects/postag_dict_all+VBN.p',
                 syllables_file='saved_objects/cmudict-0.7b.txt',
                 wv_file='saved_objects/word2vec/model.txt',
                 top_file='saved_objects/words/top_words.txt' ,
                 extra_stress_file='saved_objects/edwins_extra_stresses.txt', prompt=False):
        with open(postag_file, 'rb') as f:
            postag_dict = pickle.load(f)
        self.pos_to_words = postag_dict[1]
        self.words_to_pos = postag_dict[2]
        #these are the hardcoded scenery words
        self.pos_to_words["scNNS"] = ["forests", "branches", "roots", "twigs", "brambles", "fruits", "thorns", "bushes", "trees", "foliages", "stumps", "weeds"]
        self.pos_to_words["scJJS"] = ["darkest", "deepest", "strangest", "longest", "silentest", "quietest", "mysteriousest", "lonliest", "saddest", "eeriest"]
        self.pos_to_words["scJJ"] = ["dark", "deep", "strange", "long", "silent", "quiet",
                                      "mysterious", "lonely", "sad", "eerie"]



        for item in self.pos_to_words["scNNS"]: #we need to update the pos_to_words dictionary for each word
            if item not in self.words_to_pos.keys():
                self.words_to_pos[item] = ["scNNS"]
            else:
                self.words_to_pos[item].append("scNNS")
        for item in self.pos_to_words["scJJS"]:
            if item not in self.words_to_pos.keys():
                self.words_to_pos[item] = ["scJJS"]
            else:
                self.words_to_pos[item].append("scJJS")
        for item in self.pos_to_words["scJJ"]:
            if item not in self.words_to_pos.keys():
                self.words_to_pos[item] = ["scJJ"]
            else:
                self.words_to_pos[item].append("scJJ")

        self.special_words = helper.get_finer_pos_words()

        self.api_url = 'https://api.datamuse.com/words'

        with open(top_file) as tf:
            self.top_common_words = [line.strip() for line in tf.readlines()]


        with open("saved_objects/filtered_nouns_verbs.txt", "r") as hf:
            self.filtered_nouns_verbs = [line.strip() for line in hf.readlines()]
            self.filtered_nouns_verbs += self.pos_to_words["IN"] + self.pos_to_words["PRP"]

        self.dict_meters = helper.create_syll_dict([syllables_file], extra_stress_file)
        """try:
            with open("saved_objects/w2v.p", "rb") as pickle_in:
                self.poetic_vectors = pickle.load(pickle_in)
        except:
            print("loading vectors....")
            with open("saved_objects/w2v.p", "wb") as pickle_in:
                #self.poetic_vectors = KeyedVectors.load_word2vec_format(wv_file, binary=False)
                self.poetic_vectors = self.filtered_nouns_verbs
                pickle.dump(self.poetic_vectors, pickle_in)
                print("loaded")"""

        #print(self.poetic_vectors.shape)

        #with open("poems/shakespeare_tagged.p", "rb") as pickle_in:
        #   self.templates = pickle.load(pickle_in)
            #print(len(self.templates))

        with open("poems/end_pos.txt", "r") as pickin:
            list_pos = pickin.readlines()
            self.end_pos = {}
            for l in list_pos:
                self.end_pos[l.split()[0]] = l.split()[1:]
            #self.end_pos['NNP'] = []

        self.pos_syllables = helper.create_pos_syllables(self.pos_to_words, self.dict_meters)

        with open("saved_objects/pos_sylls_mode.p", "rb") as pickle_in:
            self.pos_sylls_mode = pickle.load(pickle_in)

        #with open("saved_objects/template_to_line.pickle", "rb") as pickle_in:
         #   self.templates = list(pickle.load(pickle_in).keys()) #update with sonnet ones one day

        #with open("saved_objects/template_no_punc.pickle", "rb") as pickle_in:
         #   self.templates = pickle.load(pickle_in)

        with open("poems/ch_template.txt", "r") as templs: #changed
            self.templates = {}
            self.tempnums = {}
            lines = templs.readlines()
            count = 1
            for line in lines:
                self.templates[" ".join(line.split()[:-1])] = line.split()[-1].strip()
                self.tempnums[count] = " ".join(line.split()[:-1])
                count += 1


        #self.character = self.gen_character("sun", "male", "radiant")
        self.character = self.gen_character("forest", "none", "savage")

        self.pos_to_words["chNN"] = list(self.character.char_words)
        self.pos_to_words["chPRP"] = [pronoun for pronoun in self.pos_to_words["PRP"] if pronoun in list(self.character.pronouns)]
        self.pos_to_words["chPRP$"] = [pronoun for pronoun in self.pos_to_words["PRP$"] if pronoun in list(self.character.pronouns)]
        self.pos_to_words["chJJ"] = list(self.character.char_adj)

        for item in self.pos_to_words["chNN"]: #we need to update the pos_to_words dictionary for each word
            if item not in self.words_to_pos.keys():
                self.words_to_pos[item] = ["chNN"]
            else:
                self.words_to_pos[item].append("chNN")

        for item in self.pos_to_words["chPRP"]:
            if item not in self.words_to_pos.keys():
                self.words_to_pos[item] = ["chPRP"]
            else:
                self.words_to_pos[item].append("chPRP")
        for item in self.pos_to_words["chPRP$"]:
            if item not in self.words_to_pos.keys():
                self.words_to_pos[item] = ["chPRP$"]
            else:
                self.words_to_pos[item].append("chPRP$")
        for item in self.pos_to_words["chJJ"]:
            if item not in self.words_to_pos.keys():
                self.words_to_pos[item] = ["chJJ"]
            else:
                self.words_to_pos[item].append("chJJ")

        if prompt:
            self.gen_poem_scenic(prompt)

    def gen_poem_scenic(self, prompt, print_poem=True):
        """

        Parameters
        ----------
        prompt - the word the base the poem on
        print_poem - optional parameter to print output

        Returns - a sonnet
        -------
        1. generate a rhyme set
        2. For every line pick a random word from the set:
            a. Get a random template which ends with the POS and meter of that word
            b. Get a random word which fits the POS and meter of the next word (working backwards)
            c. Repeat until template finished
        3. Repeat for 14 lines

        """
        #Get rhyming words
        #at some point implement narrative trajectory stuff
        rhyme_dict = {}
        #tone = ['good','good', 'good', 'good', 'bad', 'bad', 'excellent'] #for example
        #for i,j in zip(['A', 'B', 'C', 'D', 'E', 'F', 'G'], tone):
        #    rhyme_dict[i] = self.getRhymes([prompt,j]) #one day pass [prompt, narr]
        last_word_dict_complete = False
        while not last_word_dict_complete:
            for i in ['A', 'B']:
                rhyme_dict[i] = self.getRhymes([prompt], words=self.filtered_nouns_verbs)
            last_word_dict = self.last_word_dict(rhyme_dict)
            last_word_dict_complete = True
            for key in last_word_dict.keys():
                if len(last_word_dict[key]) == 0:
                    last_word_dict_complete = False
        #for now we shall generate random words, but they will fit the meter, rhyme and templates

        candidates = ["         ----" + prompt.upper() + "----"]
        used_templates = []

        for line_number in range(1, 5):
            template = self.tempnums[line_number]
            pos_needed = template.split()
            first_word = random.choice(list(last_word_dict[line_number]))  # last word is decided in last_word_dict

            #while first_word not in self.dict_meters.keys() or not self.suitable_last_word(first_word): #make sure its valid
            while not self.suitable_last_word(first_word, template):#changed valid check to ignore meter
                first_word = random.choice(list(last_word_dict[line_number]))


            in_template = self.get_word_pos(first_word)[0]

            while in_template != self.tempnums[line_number].split()[-1]:
                in_template = random.choice(self.get_word_pos(first_word)) #some words have multiple POS so make sure it picks the one with an existing template

            curr_line = first_word

            while len(curr_line.split()) < len(template.split()): #iterates until line is complete
                #if reset: print("HI", curr_line.text)

                """
                while not template:
                    print("no template", curr_line.pos_template, curr_line.text)
                    first_w = curr_line.text.split()[0]
                    first_pos = self.get_word_pos(first_w)

                    if len(first_pos) > 1:
                        curr_line.pos_template = random.choice(first_pos) + curr_line.pos_template[len(curr_line.pos_template.split()[0]):]
                        template = self.get_random_template(curr_line.pos_template, curr_line.meter)

                    else:
                        print("unfixable")
                        print(1/0)
                """

                next_pos = pos_needed[-len(curr_line.split()) - 1] #gets next POS from the right
                poss_words = self.get_pos_words(next_pos) #gets all possible words which fit pos and NOT meter

                if not poss_words:
                    print("no words", next_pos, template)
                    print(1/0) #if there arent, die

                next_word = random.choice(poss_words) #pick word randomly
                while next_word not in self.dict_meters.keys():
                    next_word = random.choice(poss_words)

                curr_line = next_word + " " + curr_line #updates line
                #template = False #make a parameter?

            #line finished generating
            print("adding line", line_number)
            print(curr_line)
            candidates.append(curr_line)

        #poem finished generating
        if print_poem:
            print("")
            print(candidates[0])
            del candidates[0]
            for cand in range(len(candidates)):
                print(candidates[cand])#, ": ", candidates[cand].meter)
                if( (cand + 1) % 4 == 0): print("")
        #return candidates


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
            words = helper.get_similar_word_henry(theme, n_return=20, word_set=set(words))
            w_rhyme_dict = {w3: {word for word in helper.get_rhyming_words_one_step_henry(self.api_url, w3) if
                                   word in self.filtered_nouns_verbs and word in self.dict_meters.keys() and word not in self.top_common_words[:70]} for #deleted: and self.filter_common_word_henry(word, fast=True)
                              w3 in words if w3 not in self.top_common_words[:70] and w3 in self.dict_meters.keys()}

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
        last_word_dict={}


        """for i in range(1,15):
            temp = []
            if i in [1,2,5,6,9,10,13]: #lines with a new rhyme
                for k in rhyme_dict[scheme[i]].keys():
                    temp.append(k)
            if i in [3,4,7,8,11,12,14]: #lines with an old line
                for k in rhyme_dict[scheme[i]].keys():
                    temp += rhyme_dict[scheme[i]][k]
            #last_word_dict[i]=[*{*temp}]
            last_word_dict[i] = temp"""
        first_rhymes = []
        for i in range(1,5):
            if i in [1, 2]:  # lines with a new rhyme -> pick a random key
                tried = set([])
                last_word_dict[i] = [random.choice(list(rhyme_dict[scheme[i]].keys()))] #NB ensure it doesnt pick the same as another one
                while not self.suitable_last_word(last_word_dict[i][0], self.tempnums[i]) or last_word_dict[i][0] in first_rhymes or any(rhyme_dict['A'][last_word_dict[i][0]] in rhyme_dict['A'][word] for word in first_rhymes):
                    last_word_dict[i] = [random.choice(list(rhyme_dict[scheme[i]].keys()))]
                first_rhymes.append(last_word_dict[i][0])
            if i in [3, 4]:  # lines with an old rhyme -> pick a random value corresponding to key of rhyming couplet
                letter = scheme[i]
                pair = last_word_dict[i-2][0]
                last_word_dict[i] = [word for word in rhyme_dict[letter][pair] if self.suitable_last_word(word, self.tempnums[i])]

        return last_word_dict

    def suitable_last_word(self, word, template): #checks if word has the part of speech of the last word in the template
        needed_pos = template.split()[-1]
        return any([w == needed_pos for w in self.words_to_pos[word]])

    def gen_character(self, word, pronouns = None, alignment = None):

        if not alignment:
            alignment = random.choice("good", "bad")
        if not pronouns:
            pronouns = random.choice("male", "female")

        new_character = character_gen.Character(word, pronouns, alignment)

        return new_character

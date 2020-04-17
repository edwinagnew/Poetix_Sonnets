import numpy as np
import pickle
from collections import defaultdict
import helper
import random
import line



#Based off limericks.py

class Sonnet_Gen():
    def __init__(self,postag_file='saved_objects/postag_dict_all+VBN.p',
                 syllables_file='saved_objects/cmudict-0.7b.txt',
                 wv_file='saved_objects/word2vec/model.txt',
                 top_file='saved_objects/words/top_words.txt' ,
                 extra_stress_file='saved_objects/edwins_extra_stresses.txt',prompt=False):
        with open(postag_file, 'rb') as f:
            postag_dict = pickle.load(f)
        self.pos_to_words = postag_dict[1]
        self.words_to_pos = postag_dict[2]

        self.special_words = helper.get_finer_pos_words()

        self.api_url = 'https://api.datamuse.com/words'

        with open(top_file) as tf:
            self.top_common_words = [line.strip() for line in tf.readlines()]


        with open("saved_objects/filtered_nouns_verbs.txt", "r") as hf:
            self.filtered_nouns_verbs = [line.strip() for line in hf.readlines()]
            self.filtered_nouns_verbs += self.pos_to_words["IN"] + self.pos_to_words["PRP"]

        self.dict_meters = helper.create_syll_dict([syllables_file], extra_stress_file)
        try:
            with open("saved_objects/w2v.p", "rb") as pickle_in:
                self.poetic_vectors = pickle.load(pickle_in)
        except:
            print("loading vectors....")
            with open("saved_objects/w2v.p", "wb") as pickle_in:
                #self.poetic_vectors = KeyedVectors.load_word2vec_format(wv_file, binary=False)
                self.poetic_vectors = self.filtered_nouns_verbs
                pickle.dump(self.poetic_vectors, pickle_in)
                print("loaded")

        #print(self.poetic_vectors.shape)

        #with open("poems/shakespeare_tagged.p", "rb") as pickle_in:
        #   self.templates = pickle.load(pickle_in)
            #print(len(self.templates))

        with open("poems/end_pos.txt", "r") as pickin:
            list = pickin.readlines()
            self.end_pos = {}
            for l in list:
                self.end_pos[l.split()[0]] = l.split()[1:]
            #self.end_pos['NNP'] = []

        self.pos_syllables = helper.create_pos_syllables(self.pos_to_words, self.dict_meters)

        with open("saved_objects/pos_sylls_mode.p", "rb") as pickle_in:
            self.pos_sylls_mode = pickle.load(pickle_in)

        #with open("saved_objects/template_to_line.pickle", "rb") as pickle_in:
         #   self.templates = list(pickle.load(pickle_in).keys()) #update with sonnet ones one day

        #with open("saved_objects/template_no_punc.pickle", "rb") as pickle_in:
         #   self.templates = pickle.load(pickle_in)

        with open("poems/shakespeare_templates.txt", "r") as templs:
            self.templates = {}
            lines = templs.readlines()
            for line in lines:
                self.templates[" ".join(line.split()[:-1])] = line.split()[-1].strip()

        """with open("saved_objects/loop_counts.txt", "r") as l_c:
            self.max_loop = int(l_c.readlines()[-1])"""

        if prompt:
            self.gen_poem_edwin(prompt)



    def gen_poem_edwin(self, prompt, print_poem=True):
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
        for i in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            rhyme_dict[i] = self.getRhymes([prompt], words=self.filtered_nouns_verbs)
        last_word_dict = self.last_word_dict(rhyme_dict)
        #candidates = self.gen_first_line_new(temp_name.lower(), search_space=5, strict=True, seed=prompt) #gonna need to think of strategy
        #for now we shall generate random words, but they will fit the meter, rhyme and templates

        candidates = ["         ----" + prompt.upper() + "----"]
        used_templates = []
        for line_number in range(1,15):
            first_word = random.choice(list(last_word_dict[line_number]))  # last word is decided
            while first_word not in self.dict_meters.keys() or not self.suitable_last_word(first_word):
                first_word = random.choice(list(last_word_dict[line_number]))
            in_template = self.get_word_pos(first_word)[0]
            while in_template not in self.end_pos or not any(pos in self.end_pos[in_template] for pos in self.dict_meters[first_word]):
                in_template = random.choice(self.get_word_pos(first_word))
            in_meter = [poss_meter for poss_meter in self.dict_meters[first_word] if poss_meter in self.end_pos[in_template]]
            if len(in_meter) < 1:
                print(first_word, in_meter)
                print(1/0)
            in_meter = in_meter[0]
            curr_line = line.Line(first_word, self.dict_meters, pos_template=in_template)
            curr_line.meter = in_meter
            curr_line.syllables = len(in_meter)
            template = False
            while curr_line.syllables < 10:
                #if reset: print("HI", curr_line.text)
                if not template:
                    template = self.get_random_template(curr_line.pos_template, curr_line.meter)
                if (line_number-1)%2 == 0 and template.split()[0] in ['AND']:
                    print("oi oi", template, line_number)
                    template = self.get_random_template(curr_line.pos_template, curr_line.meter, exclude=["AND"])
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
                if template == curr_line.pos_template:
                    #NOT GREAT
                    curr_line.syllables = 100
                    continue
                while template in used_templates:
                    template = self.get_random_template(curr_line.pos_template, curr_line.meter)
                next_pos = template.split()[-len(curr_line.pos_template.split()) - 1]
                next_meter = self.templates[template].split("_")[-len(curr_line.pos_template.split()) - 1]
                poss_words = self.get_pos_words(next_pos, meter=next_meter)
                if not poss_words:
                    print("no words", next_pos, next_meter, template)
                    print(1/0)
                next_word = random.choice(poss_words)

                #if sylls[-1] != curr_line.first_syll and curr_line.syllables + len(sylls) <= 10 and helper.isIambic(sylls) and not reset:
                curr_line.add_word(next_word, next_meter)
                curr_line.pos_template = next_pos + " " + curr_line.pos_template
                #template = False #make a parameter?

            print("adding line", line_number)
            curr_line.print_info()
            candidates.append(curr_line)
            used_templates.append(curr_line.pos_template)
        if print_poem:
            print("")
            print(candidates[0])
            del candidates[0]
            for cand in range(len(candidates)):
                print(candidates[cand].text)#, ": ", candidates[cand].meter)
                if( (cand + 1) % 4 == 0): print("")
        #return candidates


    def get_word_pos(self, word):
        """
        Get the set of POS category of a word. If we are unable to get the category, return None.
        """
        """ if "'" in word:
            if "'s" in word:
                return self.get_word_pos(word.split("'")[0]) + " " + self.get_word_pos("'")
            else:
                print("".join(word.split("'")))
                return self.get_word_pos("".join(word.split("'")))"""
        # Special case
        if word.upper() in self.special_words:
            return [word.upper()]
        if word not in self.words_to_pos:
            return None
        return self.words_to_pos[word]

    def get_pos_words(self,pos, meter=None):

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
            print("we must go deeper")
            words = helper.get_similar_word_henry(theme, n_return=20, word_set=set(words))
            w_rhyme_dict = {w3: {word for word in helper.get_rhyming_words_one_step_henry(self.api_url, w3) if
                                   word in self.poetic_vectors and word in self.dict_meters.keys() and word not in self.top_common_words[:70]} for #deleted: and self.filter_common_word_henry(word, fast=True)
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
        poss_pos: set
        contains all possible parts of speech for last word in (any) line
        Returns
        -------
        dictionary
            Format is {1: ['apple', 'orange'], 2: ['apple', orange] ... }

        """
        scheme = {1: 'A', 2: 'B', 3: 'A', 4: 'B', 5: 'C', 6: 'D', 7: 'C', 8: 'D', 9: 'E', 10: 'F', 11: 'E', 12: 'F', 13: 'G', 14: 'G'}
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
        rand_keys = {}
        #for sc in ['A', 'B', 'C', 'D', 'E', 'F', 'G']: rand_keys[sc] = random.choice(list(rhyme_dict[sc].keys()))
        first_rhymes = []
        for i in range(1,15):
            if i in [1, 2, 5, 6, 9, 10, 13]:  # lines with a new rhyme
                last_word_dict[i] = [random.choice(list(rhyme_dict[scheme[i]].keys()))] #NB ensure it doesnt pick the same as another one
                while not self.suitable_last_word(last_word_dict[i][0]) or last_word_dict[i][0] in first_rhymes or any(rhyme_dict['A'][last_word_dict[i][0]] in rhyme_dict['A'][word] for word in first_rhymes):
                    last_word_dict[i] = [random.choice(list(rhyme_dict[scheme[i]].keys()))]
                first_rhymes.append(last_word_dict[i][0])
            if i in [3, 4, 7, 8, 11, 12, 14]:  # lines with an old line
                #last_word_dict[i] = rhyme_dict[scheme[i]][rand_keys[scheme[i]]]
                letter = scheme[i]
                pair = last_word_dict[i-2][0]
                if i == 14:
                    pair = last_word_dict[13][0]
                last_word_dict[i] = [word for word in rhyme_dict[letter][pair] if self.suitable_last_word(word)]
        return last_word_dict

    def suitable_last_word(self, word): #checks pos is in self.end_pos and has correct possible meters
        return any(w in self.end_pos.keys() for w in self.get_word_pos(word)) and any(t in self.end_pos[pos] for t in self.dict_meters[word] for pos in self.get_word_pos(word) if pos in self.end_pos)


    def get_random_template(self, curr_template, curr_meter, exclude=None):
        poss_templates = [item for item in self.templates.keys() if item[-len(curr_template):] == curr_template and self.templates[item].split('_')[-len(curr_meter.split('_')):] == curr_meter.split('_')]
        if exclude: poss_templates = [x for x in poss_templates if x.split()[0] not in exclude]
        if len(poss_templates) == 0: return False
        return random.choice(poss_templates)

"""    def get_current_template(self, curr_template):
        curr = ""
        for s in curr_template.split():
            curr += self.words_to_pos[s][0] + " "

        return curr[:-1]"""

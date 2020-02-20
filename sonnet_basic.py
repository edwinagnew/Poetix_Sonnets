import numpy as np
import pickle
from collections import defaultdict
import helper
import random
import nltk


#Based off limericks.py

class Sonnet_Gen():
    def __init__(self,postag_file='saved_objects/postag_dict_all.p',
                 syllables_file='saved_objects/cmudict-0.7b.txt',
                 wv_file='saved_objects/word2vec/model.txt',
                 top_file='saved_objects/top75.txt'):
        with open(postag_file, 'rb') as f:
            postag_dict = pickle.load(f)
        self.pos_to_words = postag_dict[1]
        self.words_to_pos = postag_dict[2]

        self.api_url = 'https://api.datamuse.com/words'

        with open(top_file) as tf:
            self.top_common_words = [line.strip() for line in tf.readlines()]

        with open("saved_objects/pos_sylls_mode.p", "rb") as pickle_in:
            self.pos_sylls_mode = pickle.load(pickle_in)
            #print(self.pos_sylls_mode)

        with open("saved_objects/filtered_nouns_verbs.txt", "r") as hf:
            self.filtered_nouns_verbs = [line.strip() for line in hf.readlines()]
            self.filtered_nouns_verbs += self.pos_to_words["IN"] + self.pos_to_words["PRP"]

        self.dict_meters = helper.create_syll_dict(syllables_file)
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

        with open("poems/shakespeare_tagged.p", "rb") as pickle_in:
            self.templates = pickle.load(pickle_in)
            #print(len(self.templates))

        with open("poems/end_pos.p", "rb") as pickin:
            self.end_pos = pickle.load(pickin)

        self.pos_syllables = helper.create_pos_syllables(self.pos_to_words, self.dict_meters)



    def gen_poem_edwin(self, prompt, print_poem=True, search_space=5, retain_space=2, word_embedding_coefficient=0,stress=True, prob_threshold=-10):
        """
        Generate poems with multiple templat es given a seed word (prompt) and GPT2
        search space.
        Parameters
        ----------
        prompt: str
            A seed word used to kickstart poetry generation.
        search_space : int
            Search space of the sentence finding algorithm.
            The larger the search space, the more sentences the network runs
            in parallel to find the best one with the highest score.
        retain_space : int
            How many sentences per template to keep.
        stress: bool
            Whether we enforce stress.
        prob_threshold: float
            If the probability of a word is lower than this threshold we will not consider
            this word. Set it to None to get rid of it.
        """
        #Get rhyming words
        #at some point implement narrative trajectory stuff
        rhyme_dict = {}
        tone = ['good','good', 'good', 'good', 'bad', 'bad', 'excellent'] #for example
        for i,j in zip(['A', 'B', 'C', 'D', 'E', 'F', 'G'], tone):
            rhyme_dict[i] = self.getRhymes([prompt,j]) #one day pass [prompt, narr]

        last_word_dict = self.last_word_dict(rhyme_dict, self.end_pos)
        #print(last_word_dict)

        #candidates = self.gen_first_line_new(temp_name.lower(), search_space=5, strict=True, seed=prompt) #gonna need to think of strategy
        #for now we shall generate random words, but they will fit the meter, rhyme and templates

        candidates = []
        for line in range(1,15):
            text = random.choice(list(last_word_dict[line])) #last word is decided
            if text not in self.dict_meters.keys(): text = random.choice(list(last_word_dict[line])) #make a better fix than this
            meter = self.dict_meters[text][0]
            syllables = len(meter)
            prev_emph = meter[0]
            curr_template = [self.words_to_pos[text][0]]
            poss_templates = self.templates
            while syllables < 10:
                #word = self.filtered_nouns_verbs[random.randint(0,len(self.filtered_nouns_verbs) - 1)]
                word = random.choice(self.filtered_nouns_verbs)
                if word not in self.dict_meters.keys(): continue
                sylls = self.dict_meters[word][0]
                if sylls[-1] != prev_emph and syllables + len(sylls) <= 10 and helper.isIambic(sylls) and self.hasTemplate([self.words_to_pos[word][0]] + curr_template, threshold=1):
                    #print(word, sylls)
                    #print("accepted")
                    if syllables + len(sylls) == 9: continue #try to prevent every line beginning with in
                    text = word + " " + text
                    syllables += len(sylls)
                    meter = sylls + meter
                    prev_emph = sylls[0]
                    curr_template = [self.words_to_pos[word][0]] + curr_template
                    #print(curr_template)
            #print(text, ":", syllables, ",",meter)
            candidates.append(text)
        if print_poem:
            print("")
            print("   " , prompt, " - a computer \n")
            for cand in range(len(candidates)):
                print(candidates[cand])
                if( (cand + 1) % 4 == 0): print("")
        return candidates


    def getRhymes(self, theme):
        """
        :param theme: an array of either [prompt] or [prompt, line_theme] to find similar words to. JUST PROMPT FOR NOW
        :return: all words which rhyme with similar words to the theme in format {similar word: [rhyming words], similar word: [rhyming words], etc.}
        """
        prompt = theme[0]
        tone = theme[1]
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
            words = helper.get_similar_word_henry(theme, n_return=20, word_set=set(self.filtered_nouns_verbs))
            w_rhyme_dict = {w3: {word for word in helper.get_rhyming_words_one_step_henry(self.api_url, w3) if
                                   word in self.poetic_vectors and word not in self.top_common_words} for #deleted: and self.filter_common_word_henry(word, fast=True)
                              w3 in words if w3 not in self.top_common_words}

            #if len(w_rhyme_dict) > 0:
            mydict[prompt][tone] = {k: v for k, v in w_rhyme_dict.items() if len(v) > 0}

        with open("saved_objects/saved_rhymes", "wb") as pickle_in:
            pickle.dump(mydict, pickle_in)
        return mydict[prompt][tone]

    def last_word_dict(self, rhyme_dict, poss_pos):
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
                if nltk.pos_tag(last_word_dict[i])[0][1] not in poss_pos or last_word_dict[i][0] in first_rhymes:
                    i-=1
                    continue
                first_rhymes.append(last_word_dict[i][0])
            if i in [3, 4, 7, 8, 11, 12, 14]:  # lines with an old line
                #last_word_dict[i] = rhyme_dict[scheme[i]][rand_keys[scheme[i]]]
                letter = scheme[i]
                pair = last_word_dict[i-2][0]
                if i == 14:
                    pair = last_word_dict[13][0]
                last_word_dict[i] = [word for word in rhyme_dict[letter][pair] if nltk.pos_tag([word])[0][1] in poss_pos]
        return last_word_dict


    def hasTemplate(self, stem, threshold=5):
        #TODO - webscrape https://www.poetryfoundation.org/poems/browse#page=1&sort_by=recently_added&forms=263 to get more templates
        #consider choosing from remaining availabe templates rather than hoping to randomly stumble across one?
        #sub_templates = [item[-len(stem):] for item in self.templates]
        sub_templates = [item for item in self.templates if item[-len(stem):] == stem]
        #print("has template", stem, sub_templates, len(sub_templates))
        #return len(sub_templates) > threshold
        return True

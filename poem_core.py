from py_files import helper

import random
import pickle
import numpy as np

import string
import pronouncing
#import gpt_2_gen

from os import path

class Poem:
    def __init__(self, words_file="saved_objects/tagged_words.p",
                 templates_file='poems/jordan_templates.txt',
                 syllables_file='saved_objects/cmudict-0.7b.txt',
                 extra_stress_file='saved_objects/edwins_extra_stresses.txt',
                 top_file='saved_objects/words/top_words.txt',
                 mistakes_file=None):
        while mistakes_file and not path.exists(mistakes_file): mistakes_file = input(mistakes_file + "does not exist on your laptop, please enter your path now and/or when creating a poem object or change the code (ask edwin): ")
        keep_scores = "byron" in words_file
        self.pos_to_words, self.words_to_pos = helper.get_new_pos_dict(words_file, mistakes_file=mistakes_file, keep_scores=keep_scores)
        self.backup_words = None
        try:
            with open(templates_file) as tf:
                self.templates = [(" ".join(line.split()[:-1]), line.split()[-1]) for line in tf.readlines() if "#" not in line and len(line) > 1]
        except:
            print(templates_file, " does not exist so reading from poems/jordan_templates.txt instead")
            with open("poems/jordan_templates.txt") as tf:
                self.templates = [(" ".join(line.split()[:-1]), line.split()[-1]) for line in tf.readlines() if "#" not in line and len(line) > 1]

        self.special_words = helper.get_finer_pos_words()

        self.dict_meters = helper.create_syll_dict([syllables_file], extra_stress_file)

        self.pron = {w.split()[0].lower(): " ".join(w.split()[1:]) for w in open(syllables_file).readlines() if w.split()[0].lower().split("(")[0] in self.words_to_pos}


        with open(top_file) as tf:
            self.top_common_words = [line.strip() for line in tf.readlines()][:125]


        places = open("saved_objects/words/places.txt").readlines()
        self.pos_to_words["PLC"] = {p.strip(): 1 for p in places}
        for p in places:
            p = p.strip()
            if p not in self.words_to_pos: self.words_to_pos[p] = []
            self.words_to_pos[p].append("PLC")

        names = open("saved_objects/words/names.txt").readlines()
        curr = None
        self.all_names = {}
        for n in names:
            n = n.strip()
            if "#" in n:
                curr = n.split("#")[-1]
                continue
            if curr not in self.all_names: self.all_names[curr] = []
            self.all_names[curr].append(n)

        self.reset_gender()

        self.api_url = 'https://api.datamuse.com/words'

        self.gpt = None

    def get_meter(self, word):
        if word[-1] in ".,?;":
            return self.get_meter(word[:-1])
        elif word[-1] == ">":
            return self.get_meter(word.split("<")[0])
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

    def get_pos_words(self,pos, meter=None, rhyme=None):
        """
        Gets all the words of a given POS
        Parameters
        ----------
        pos - the POS you want
        meter - (optional) returns only words which fit the given meter, e.g. 101
        """
        if rhyme: return [w for w in self.get_pos_words(pos, meter=meter) if w in self.rhymes(w, rhyme)]
        #print("oi," , pos, meter, phrase)
        punc = [".", ",", ";", "?", ">"]
        #print("here2", pos, meter)
        if pos[-1] in punc:
            p = pos[-1]
            if p == ">":
                p = random.choice(pos.split("<")[-1].strip(">").split("/"))
                pos = pos.split("<")[0] + p
            return [word + p for word in self.get_pos_words(pos[:-1], meter=meter)]
        if pos in self.special_words:
            return [pos.lower()]
        if "PRP" in pos and "_" not in pos and meter:
            ret = [p for p in self.pos_to_words[pos] if p in self.gender and any(len(meter) == len(q) for q in self.get_meter(p)) ]
            #if len(ret) == 0: ret = [input("PRP not happening " + pos + " '" + meter + "' " + str(self.gender) + str([self.dict_meters[p] for p in self.gender]))]
            if len(ret) == 0: return [p for p in self.pos_to_words[pos] if meter in self.get_meter(p)]
            return ret
        elif pos not in self.pos_to_words:
            return []
        if meter:
            ret = [word for word in self.pos_to_words[pos] if word in self.dict_meters and meter in self.dict_meters[word]]
            return ret
        return [p for p in self.pos_to_words[pos]]

    def can_rhyme(self, pair1, pair2):
        """
        pair1 - (pos, meter)
        pair2 - (pos, meter)

        Returns - whether it is possible the two words to rhyme
        """
        if not pair1 or not pair2 or not any(pair1) or not any(pair2): return False
        set1 = set(self.get_pos_words(pair1[0], pair1[1]))
        set2 = set(self.get_pos_words(pair2[0], pair2[1]))
        return any(r1 in set2 for w1 in set1 for r1 in self.get_rhyme_words(w1)) or any(r2 in set1 for w2 in set2 for r2 in self.get_rhyme_words(w2))

    def rhymes(self, word1, word2, check_cmu=False):
        if not word1 or not word2: return False
        if word1[-1] in ".,?!>": word1 = word1.translate(str.maketrans('', '', string.punctuation))
        if word2[-1] in ".,?!>": word2 = word2.translate(str.maketrans('', '', string.punctuation))
        if word1 in self.get_rhyme_words(word2) or word2 in self.get_rhyme_words(word1): return True
        if not check_cmu: return False

        def rhyming_syll(pron):
            found_one = False
            for i in range(len(pron)-1, 0, -1):
                if pron[i] == "1": found_one = True
                if found_one and pron[i] == " ": return pron[i+1:]

        #if rhyming_syll(self.pron[word1]) == rhyming_syll(self.pron[word2]): return True
        for j in range(4):
            w1 = (word1 + "(" + str(j) + ")").replace("(0)", "")
            if w1 in self.pron:
                for k in range(4):
                    w2 = (word2 + "(" + str(k) + ")").replace("(0)", "")
                    if w2 in self.pron:
                        if rhyming_syll(self.pron[w1]) == rhyming_syll(self.pron[w2]): return True

        return False

    def get_rhyme_words(self, word):
        if not word: return []
        if word[-1] in string.punctuation:
            word = word.translate(str.maketrans('', '', string.punctuation))
            return self.get_rhyme_words(word)
        return pronouncing.rhymes(word)

    def weighted_choice(self,pos, meter=None, rhyme=None):
        punc = ".,;?!"
        if pos[-1] == ">": return self.weighted_choice(pos.split("<")[0], meter=meter, rhyme=rhyme) + random.choice(pos.split("<")[-1].strip(">").split("/"))
        if pos[-1] in punc: return self.weighted_choice(pos[:-1], meter=meter, rhyme=rhyme) + pos[-1]
        poss = self.get_pos_words(pos, meter=meter, rhyme=rhyme)
        if not poss:
            print("nope", pos, meter, rhyme)
            return ""
        elif len(poss) == 1: return poss[0]
        poss_dict = {p:self.pos_to_words[pos][p] for p in poss}
        vals = list(poss_dict.values())
        if len(vals) < 2 or min(vals) == max(vals): return random.choice(poss)
        else:
            word = np.random.choice(poss, p=helper.softmax(vals))
            self.pos_to_words[pos][word] /= 2
            return word

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
                #print(i, last_word_dict[i-2])
                if i == 14:
                    pair = last_word_dict[13][0]
                else:
                    pair = last_word_dict[i - 2][0]
                print(i, letter, pair, rhyme_dict[letter][pair])
                last_word_dict[i] = [word for word in rhyme_dict[letter][pair] if self.suitable_last_word(word)]
                if len(last_word_dict[i]) == 0:
                    print("not happening")
                    return self.last_word_dict(rhyme_dict)
        return last_word_dict

    def suitable_last_word(self, word, punc=False): #checks pos is in self.end_pos and has correct possible meters
        if punc: return self.suitable_last_word(word + ".") or self.suitable_last_word(word + "?")
        return any(w in self.end_pos for w in self.get_word_pos(word)) and any(t in self.end_pos[pos] for t in self.dict_meters[word] for pos in self.get_word_pos(word) if pos in self.end_pos)

    def write_line_gpt(self, template, meter, rhyme_word=None, n=1, gpt_model=None, verbose=False):
        if not self.gpt:
            #self.gpt = gpt_2_gen.gpt(seed=None, sonnet_method=self.get_pos_words)
            self.gpt = gpt_model
            if not gpt_model: print("need a gpt model", 1/0)
        print("\n")
        if "he" in self.gender or "she" in self.gender:
            template = template.replace("VBP", "VBZ").replace("DO", "DOES")
        else:
            template = template.replace("VBZ", "VBP").replace("DOES", "DO")

        print(template, meter)
        for i in range(n):
            #print("generating with ", t_2, meter.split("_"), i)
            print(self.gpt.good_generation(template=template.split(), meter=meter.split("_"), rhyme_word=rhyme_word, verbose=verbose))

    def write_line_random(self, template, meter, rhyme_word=None, n=1):
        print("writing line", template, meter)
        if rhyme_word and type(rhyme_word) == list: rhyme_word = rhyme_word[-1]
        if rhyme_word: print("rhyme word:", rhyme_word)
        if type(template) == str: template = template.split()
        if type(meter) == str: meter = meter.split("_")

        if "he" in self.gender or "she" in self.gender:
            template = template.replace("VBP", "VBZ").replace("DO", "DOES")
        else:
            template = template.replace("VBZ", "VBP").replace("DOES", "DO")



        if n > 1: return [self.write_line_random(template, meter, rhyme_word) for i in range(n)]

        line = ""
        punc = ",.;?"


        for i in range(len(template)):
            next_word = self.weighted_choice(template[i], meter[i])
            if not next_word: input("no word for " + template[i] + meter[i])
            space = " " * int(line != "" and next_word not in (punc + "'s"))
            line += space + next_word


        new_word = ""
        while rhyme_word and not self.rhymes(new_word, rhyme_word):
            print("trying to rhyme", template[-1], meter[-1], new_word, "with", rhyme_word)
            old_word = line.split()[-1].translate(str.maketrans('', '', string.punctuation))
            self.reset_letter_words()
            new_word = self.weighted_choice(template[-1], meter[-1], rhyme=rhyme_word).translate(str.maketrans('', '', string.punctuation))
            print("got", new_word)
            if not new_word:
                print("cant rhyme")
                return 1/0
            line = line.replace(old_word, new_word) #will replace all instances



        return line.strip()

    def reset_letter_words(self):
        for pos in list(self.pos_to_words):
            if "_" in pos: #or pos in "0123456789"
                del self.pos_to_words[pos]

    def reset_gender(self):
        self.gender = random.choice([["i", "me", "my", "mine", "myself"], ["you", "your", "yours", "yourself"],  ["he", "him", "his", "himself"], ["she", "her", "hers", "herself"], ["we", "us", "our", "ours", "ourselves"], ["they", "them", "their", "theirs", "themselves"]])

        g = random.choice(["male", "female"])
        if "he" in self.gender: g = "male"
        elif "she" in self.gender: g = "female"

        self.pos_to_words["NAM"] = {n: 1 for n in self.all_names[g]}


    def get_next_template(self, used_templates, check_the_rhyme=None):
        poss = self.templates
        incomplete = ",;" + string.ascii_lowercase
        n = len(used_templates)
        if n > 0:
            if used_templates[-1][-1] == ".":
                poss = [p for p in poss if p[0].split()[0] not in ["AND", "THAT", "OR", "SHALL", "WILL", "WHOSE"]]
            #elif used_templates[-1][-1] in incomplete:
             #   poss = [p.replace("?", ".") for p in poss if p[0].split()]

            if n % 4 == 3 or n == 13:
                poss = [(p.replace("/,", ""), q) for p,q in poss if p[-1] in ">.?"]
                #print("last line of stanza so:", poss)

        if n % 4 == 0:
            poss = [(p, q) for p, q in poss if p.split()[0] not in ["AND", "OR"]]


        if len(poss) == 0:
            print("theres no templates " + str(len(used_templates)) + used_templates[-1])
            return random.choice(self.templates)
        if "he" in self.gender or "she" in self.gender:
            poss = [(p[0].replace("VBP", "VBZ"), p[1]) for p in poss]
        else:
            poss = [(p[0].replace("VBZ", "VBP"), p[1]) for p in poss]

        if check_the_rhyme: poss = [p for p in poss if any(self.rhymes(check_the_rhyme, w) for w in self.get_pos_words(p[0].split()[-1], p[1].split("_")[-1]))]
        t = random.choice(poss)
        if "<" in t: t = t.split("<")[0] + random.choice(t.split("<")[-1].strip(">").split("/"))
        return t



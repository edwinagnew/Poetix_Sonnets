from py_files import helper

import random
import pickle
import numpy as np

import string
import pronouncing
# import gpt_2_gen

from os import path


class Poem:
    def __init__(self, words_file="saved_objects/tagged_words.p",
                 templates_file=('poems/jordan_templates.txt', "poems/rhetorical_templates.txt"),
                 syllables_file='saved_objects/cmudict-0.7b.txt',
                 extra_stress_file='saved_objects/edwins_extra_stresses.txt',
                 top_file='saved_objects/words/top_words.txt',
                 mistakes_file=None):
        while mistakes_file and not path.exists(mistakes_file): mistakes_file = input(
            mistakes_file + "does not exist on your laptop, please enter your path now and/or when creating a poem object or change the code (ask edwin): ")
        keep_scores = False  # "byron" in words_file
        self.pos_to_words, self.words_to_pos = helper.get_new_pos_dict(words_file, mistakes_file=mistakes_file,
                                                                       keep_scores=keep_scores)
        self.backup_words = None
        if "byron" in words_file:
            self.backup_words, _ = helper.get_new_pos_dict('saved_objects/tagged_words.p')
        if type(templates_file) == str: templates_file = [templates_file]
        self.templates = []
        for t in templates_file:
            try:
                with open(t) as tf:
                    self.templates += [(" ".join(line.split()[:-1]), line.split()[-1]) for line in tf.readlines() if
                                       "#" not in line and len(line) > 1]
            except:
                print(t, " does not exist so reading from poems/jordan_templates.txt instead")
                with open("poems/jordan_templates.txt") as tf:
                    self.templates = [(" ".join(line.split()[:-1]), line.split()[-1]) for line in tf.readlines() if
                                      "#" not in line and len(line) > 1]

        self.special_words = helper.get_finer_pos_words()

        self.dict_meters = helper.create_syll_dict([syllables_file], extra_stress_file)

        self.pron = {w.split()[0].lower(): " ".join(w.split()[1:]) for w in open(syllables_file).readlines() if
                     w.split()[0].lower().split("(")[0] in self.words_to_pos}

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

        self.possible_meters = ["1", "0", "10", "01", "101", "010", "1010", "0101", "10101", "01010", "101010",
                                "010101"]  # the possible meters a word could have

        self.gender = []

        self.set_meter_pos_dict()
        self.reset_gender()

        self.api_url = 'https://api.datamuse.com/words'

        self.gpt = None
        self.gpt_past = ""

    def get_meter(self, word):
        if not word or len(word) == 0: return [""]
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
        word = helper.remove_punc(word)
        if word.upper() in self.special_words:
            return [word.upper()] + (self.words_to_pos[word] if word in self.words_to_pos else [])
        if word not in self.words_to_pos:
            return []
        return self.words_to_pos[word]

    def get_pos_words(self, pos, meter=None, rhyme=None):
        """
        Gets all the words of a given POS
        Parameters
        ----------
        pos - the POS you want
        meter - (optional) returns only words which fit the given meter, e.g. 101
        """
        if rhyme: return [w for w in self.get_pos_words(pos, meter=meter) if self.rhymes(w, rhyme)]
        # print("oi," , pos, meter, phrase)
        punc = [".", ",", ";", "?", ">"]
        # print("here2", pos, meter)
        if pos[-1] in punc:
            p = pos[-1]
            if p == ">":
                p = random.choice(pos.split("<")[-1].strip(">").split("/"))
                pos = pos.split("<")[0] + p
            return [word + p for word in self.get_pos_words(pos[:-1], meter=meter)]
        if pos in self.special_words:
            return [pos.lower()]
        if meter and type(meter) == str: meter = [meter]
        if "PRP" in pos and "_" not in pos and meter:
            ret = [p for p in self.pos_to_words[pos] if p in self.gender and any(q in meter for q in self.get_meter(p))]
            # if len(ret) == 0: ret = [input("PRP not happening " + pos + " '" + meter + "' " + str(self.gender) + str([self.dict_meters[p] for p in self.gender]))]
            # if len(ret) == 0: return [p for p in self.pos_to_words[pos] if any(m in self.get_meter(p) for m in meter)]
            return ret
        elif pos not in self.pos_to_words:
            return []
        if meter:
            ret = [word for word in self.pos_to_words[pos] if
                   word in self.dict_meters and any(m in self.dict_meters[word] for m in meter)]
            return ret
        return [p for p in self.pos_to_words[pos]]

    def get_backup_pos_words(self, pos, meter=None, rhyme=None):
        if not self.backup_words: return []
        temp = self.pos_to_words
        self.pos_to_words = self.backup_words
        words = self.get_pos_words(pos, meter, rhyme)
        self.pos_to_words = temp
        return words

    def can_rhyme(self, pair1, pair2):
        """
        pair1 - (pos, meter)
        pair2 - (pos, meter)

        Returns - whether it is possible the two words to rhyme
        """
        if not pair1 or not pair2 or not any(pair1) or not any(pair2): return False
        set1 = set(self.get_pos_words(pair1[0], pair1[1]))
        set2 = set(self.get_pos_words(pair2[0], pair2[1]))
        return any(r1 in set2 for w1 in set1 for r1 in self.get_rhyme_words(w1)) or any(
            r2 in set1 for w2 in set2 for r2 in self.get_rhyme_words(w2))

    def rhymes(self, word1, word2, check_cmu=False):
        if not word1 or not word2: return False
        if word1[-1] in ".,?!>": word1 = word1.translate(str.maketrans('', '', string.punctuation))
        if word2[-1] in ".,?!>": word2 = word2.translate(str.maketrans('', '', string.punctuation))
        if word1 in self.get_rhyme_words(word2) or word2 in self.get_rhyme_words(word1): return True
        if not check_cmu: return False

        def rhyming_syll(pron):
            found_one = False
            for i in range(len(pron) - 1, 0, -1):
                if pron[i] == "1": found_one = True
                if found_one and pron[i] == " ": return pron[i + 1:]

        # if rhyming_syll(self.pron[word1]) == rhyming_syll(self.pron[word2]): return True
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

    def weighted_choice(self, pos, meter=None, rhyme=None):
        punc = ".,;?!"
        if pos[-1] == ">": return self.weighted_choice(pos.split("<")[0], meter=meter, rhyme=rhyme) + random.choice(
            pos.split("<")[-1].strip(">").split("/"))
        if pos[-1] in punc: return self.weighted_choice(pos[:-1], meter=meter, rhyme=rhyme) + pos[-1]
        poss = self.get_pos_words(pos, meter=meter, rhyme=rhyme)
        if not poss:
            print("nope", pos, meter, rhyme)
            return ""
        elif len(poss) == 1:
            return poss[0]
        poss_dict = {p: self.pos_to_words[pos][p] for p in poss}
        vals = list(poss_dict.values())
        if len(vals) < 2 or min(vals) == max(vals):
            return random.choice(poss)
        else:
            word = np.random.choice(poss, p=helper.softmax(vals))
            self.pos_to_words[pos][word] /= 2
            return word

    def getRhymes(self, theme, words):
        """
        :param theme: an array of either [prompt] or [prompt, line_theme] to find similar words to. JUST PROMPT FOR NOW
        :return: all words which rhyme with similar words to the theme in format {similar word: [rhyming words], similar word: [rhyming words], etc.}
        """
        # if type(theme) != list:
        #    theme = [theme]
        try:
            with open("saved_objects/saved_rhymes", "rb") as pickle_in:
                mydict = pickle.load(pickle_in)

        except:
            with open("saved_objects/saved_rhymes", "wb") as pickle_in:
                mydict = {}
                pickle.dump(mydict, pickle_in)
        if theme not in mydict.keys():
            mydict[theme] = {}
            print("havent stored anything for ", theme, "please wait...")
            print(" (ignore the warnings) ")
            words = helper.get_similar_word_henry(theme.lower().split(), n_return=40, word_set=set(words))
            w_rhyme_dict = {w3: {word for word in helper.get_rhyming_words_one_step_henry(self.api_url, w3) if
                                 word in self.words_to_pos and word in self.dict_meters and word not in self.top_common_words[
                                                                                                        :70]} for
                            # deleted: and self.filter_common_word_henry(word, fast=True)
                            w3 in words if w3 not in self.top_common_words[:70] and w3 in self.dict_meters}

            # if len(w_rhyme_dict) > 0:
            mydict[theme] = {k: v for k, v in w_rhyme_dict.items() if len(v) > 0}
        elif "NONE" in mydict[theme]:
            print("the code for this bit has changed. Please delete saved_objects/saved_rhymes and start again", 1 / 0)
        with open("saved_objects/saved_rhymes", "wb") as pickle_in:
            pickle.dump(mydict, pickle_in)
        return mydict[theme]

    def last_word_dict(self, rhyme_dict,
                       scheme={1: 'A', 2: 'B', 3: 'A', 4: 'B', 5: 'C', 6: 'D', 7: 'C', 8: 'D', 9: 'E', 10: 'F', 11: 'E',
                               12: 'F', 13: 'G', 14: 'G'}):
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
        last_word_dict = {}

        print(rhyme_dict)
        first_rhymes = []
        for i in range(1, 15):
            if i in [1, 2, 5, 6, 9, 10, 13]:  # lines with a new rhyme -> pick a random key
                last_word_dict[i] = [random.choice(
                    list(rhyme_dict[scheme[i]].keys()))]  # NB ensure it doesnt pick the same as another one
                while not self.suitable_last_word(last_word_dict[i][0]) or last_word_dict[i][0] in first_rhymes or any(
                        rhyme_dict['A'][last_word_dict[i][0]] in rhyme_dict['A'][word] for word in first_rhymes):
                    last_word_dict[i] = [random.choice(list(rhyme_dict[scheme[i]].keys()))]
                first_rhymes.append(last_word_dict[i][0])
            if i in [3, 4, 7, 8, 11, 12,
                     14]:  # lines with an old rhyme -> pick a random value corresponding to key of rhyming couplet
                letter = scheme[i]
                # print(i, last_word_dict[i-2])
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

    def suitable_last_word(self, word, punc=False):  # checks pos is in self.end_pos and has correct possible meters
        if punc: return self.suitable_last_word(word + ".") or self.suitable_last_word(word + "?")
        return any(w in self.end_pos for w in self.get_word_pos(word)) and any(
            t in self.end_pos[pos] for t in self.dict_meters[word] for pos in self.get_word_pos(word) if
            pos in self.end_pos)

    def write_line_gpt(self, template=None, meter=None, rhyme_word=None, n=1, gpt_model=None, flex_meter=False,
                       all_verbs=False, verbose=False, alliteration=None):
        if not self.gpt:
            # self.gpt = gpt_2_gen.gpt(seed=None, sonnet_method=self.get_pos_words)
            self.gpt = gpt_model
            if not gpt_model: print("need a gpt model", 1 / 0)

        if n > 1: return [self.write_line_gpt(template, meter, rhyme_word, flex_meter=flex_meter, all_verbs=all_verbs,
                                              verbose=verbose, alliteration=alliteration) for _ in range(n)]

        if template is None: template, meter = random.choice(self.templates)

        template = self.fix_template(template)
        if all_verbs: template = template.replace("VB", "*VB")

        if flex_meter:
            base_template = template.replace("*VB", "VB").replace("sc", "")
            self.check_template(base_template, meter)
            rhyme_pos = helper.remove_punc(template.split()[-1]).split("_")[-1]
            # if "_" in rhyme_pos
            if rhyme_word:
                if "__" in rhyme_word:
                    input("here" + rhyme_word)
                    r = set(self.get_meter(rhyme_word.strip("__")))
                else:
                    rhyme_words = self.get_pos_words(rhyme_pos, rhyme=rhyme_word) if type(rhyme_word) == str else rhyme_word
                    r = set([x for x in ["1", "01", "101", "0101", "10101"] for w in rhyme_words if
                             x in self.get_meter(w)]) if rhyme_word else None
                    if len(r) == 0 or len(rhyme_words) == 0:
                        if verbose: print("couldn't get a rhyme here:", template, rhyme_word, rhyme_words, r)
                        return None
            else:
                r = None

            if "_" in base_template:
                meter_dict = {}
                for m in meter.split("_")[::-1]:
                    meter_dict = {m: meter_dict.copy()}
            else:
                meter_dict = self.get_poss_meters_forward(base_template, "01" * 5, r)
            if not meter_dict:
                if verbose: print("couldn't get a meter_dict:", template, rhyme_word)
                return None
            if verbose: print("writing flexible line", template, meter_dict, rhyme_word)

            return self.gpt.generation_flex_meter(template.split(), meter_dict, seed=self.gpt_past,
                                                  rhyme_word=rhyme_word, verbose=verbose, alliteration=alliteration)

        else:
            if verbose: print("writing line", template, meter)
            # if n > 1: return [self.gpt.good_generation(template=template.split(), meter=meter.split("_"), rhyme_word=rhyme_word, verbose=verbose) for i in range(n)]

            return self.gpt.good_generation(seed=self.gpt_past, template=template.split(), meter=meter.split("_"),
                                            rhyme_word=rhyme_word, verbose=verbose)

    def write_line_random(self, template=None, meter=None, rhyme_word=None, n=1, verbose=False):
        if template is None: template, meter = random.choice(self.templates)
        template = self.fix_template(template)

        if n > 1: return [self.write_line_random(template, meter, rhyme_word) for i in range(n)]
        print("writing line", template, meter)
        if rhyme_word and type(rhyme_word) == list: rhyme_word = rhyme_word[-1]
        if rhyme_word and verbose: print("rhyme word:", rhyme_word)
        if type(template) == str: template = template.split()
        if type(meter) == str: meter = meter.split("_")

        line = ""
        punc = ",.;?"

        for i in range(len(template)):
            next_word = self.weighted_choice(template[i], meter[i])
            if not next_word: input("no word for " + template[i] + meter[i])
            space = " " * int(line != "" and next_word not in (punc + "'s"))
            line += space + next_word

        new_word = ""
        while rhyme_word and not self.rhymes(new_word, rhyme_word):
            if verbose: print("trying to rhyme", template[-1], meter[-1], new_word, "with", rhyme_word)
            old_word = line.split()[-1].translate(str.maketrans('', '', string.punctuation))
            self.reset_letter_words()
            new_word = self.weighted_choice(template[-1], meter[-1], rhyme=rhyme_word).translate(
                str.maketrans('', '', string.punctuation))
            if verbose: print("got", new_word)
            if not new_word:
                print("cant rhyme")
                return 1 / 0
            line = line.replace(old_word, new_word)  # will replace all instances

        return line.strip()

    def reset_letter_words(self):
        for pos in list(self.pos_to_words):
            if "_" in pos:  # or pos in "0123456789"
                del self.pos_to_words[pos]

    def reset_gender(self):
        self.gender = random.choice(
            [["i", "me", "my", "mine", "myself"], ["you", "your", "yours", "yourself"], ["he", "him", "his", "himself"],
             ["she", "her", "hers", "herself"], ["we", "us", "our", "ours", "ourselves"],
             ["they", "them", "their", "theirs", "themselves"]])

        g = random.choice(["male", "female"])
        if "he" in self.gender:
            g = "male"
        elif "she" in self.gender:
            g = "female"

        self.pos_to_words["NAM"] = {n: 1 for n in self.all_names[g]}

        self.reset_meter_pos_dict()

    def set_meter_pos_dict(self):
        self.meter_and_pos = {}
        possible_pos = list(self.pos_to_words.keys())
        for pos in possible_pos:
            for meter in self.possible_meters:
                if "PRP" in pos:
                    self.meter_and_pos[(meter, pos)] = [word for word in self.pos_to_words[pos] if
                                                        word in self.dict_meters and meter in self.dict_meters[
                                                            word] and word in self.gender]
                else:
                    self.meter_and_pos[(meter, pos)] = [word for word in self.pos_to_words[pos] if
                                                        word in self.dict_meters and meter in self.dict_meters[word]]
        for word in self.special_words:
            if word in self.dict_meters:
                meter = self.dict_meters[word]
                self.meter_and_pos[(meter, word)] = [word]
            else:
                continue

    def reset_meter_pos_dict(self):
        possible_pos = [item for item in list(self.pos_to_words.keys()) if "PRP" in item]
        for pos in possible_pos:
            for meter in self.possible_meters:
                self.meter_and_pos[(meter, pos)] = [word for word in self.pos_to_words[pos] if
                                                    word in self.dict_meters and meter in self.dict_meters[
                                                        word] and word in self.gender]

    def get_template_from_line(self, line, backwards=False):
        words = line if type(line) == list else line.lower().split()
        poss = self.templates
        if not backwards:
            for i, word in enumerate(words):
                poss = [p for p in poss if helper.remove_punc(p[0].split()[i]) in self.get_word_pos(word)]
                if len(poss) == 1: return poss
        else:
            for i in range(-1, -len(words) - 1, -1):
                poss = [p for p in poss if helper.remove_punc(p[0].split()[i]) in self.get_word_pos(words[i])]
                if len(poss) == 1: return poss
        return poss

    def get_next_template(self, used_templates, check_the_rhyme=None, end=""):
        """

        Parameters
        ----------
        used_templates
        check_the_rhyme
        end - makes sure the template could have that word at the end (but only if it starts with __)

        Returns
        -------

        """
        if len(used_templates) > 0 and type(used_templates[0]) == tuple: used_templates = [u[0] for u in used_templates]
        poss = self.templates
        # incomplete = ",;" + string.ascii_lowercase
        n = len(used_templates)
        if n > 0:
            if used_templates[-1][-1] in ".?":
                poss = [p for p in poss if p[0].split()[0] not in ["AND", "THAT", "OR", "SHALL", "WILL", "WHOSE", "TO", "WAS", "VBD", "IN"]]
            # elif used_templates[-1][-1] in incomplete:
            #   poss = [p.replace("?", ".") for p in poss if p[0].split()]

            if n % 4 == 3 or n == 13:
                # poss = [p for p in poss if p[0][-1] not in ",;" + string.ascii_uppercase]
                poss = [(p.replace("/,", "").replace("<,/", "<"), q) for p, q in poss if p[-1] in ">."]
                # print("last line of stanza so:", poss)

            if n % 4 == 0:
                poss = [(p, q) for p, q in poss if p.split()[0] not in ["AND", "OR"]]
            elif sum([int("_" in t) for t in used_templates]) > 1:
                poss = [(p,q) for p,q in poss if "_" not in p]


            if n % 4 > 1 or n == 13:
                poss = [(p, q) for p, q in poss if "_" not in p.split()[-1]]

        else:
            # starting templates taken from google doc
            poss = [("A JJ NN VBD IN NNS OF NN<,/.>", "0_10_10_1_0_1_0_1"),
                    ("IF PRPS COULD VB THIS JJ NN OF ABNN,", "0_1_0_10_1_0_1_0_1"),
                    ("WHAT JJ NN VBZ PRP$ NN?", "0_1010_10_1_0_1"),
                    ("PRPS VBC JJ TO VB THE NNS", "0_1_01_0_1_0_101")]

        if len(poss) == 0:
            print("theres no templates " + str(len(used_templates)) + used_templates[-1])
            return random.choice(self.templates)

        if check_the_rhyme: poss = [p for p in poss if any(
            self.rhymes(check_the_rhyme, w) for w in self.get_pos_words(p[0].split()[-1], p[1].split("_")[-1]))]
        if end and type(end) == set:
            # pos = helper.remove_punc(self.get_word_pos(end.strip("__")))
            pos = set([self.get_word_pos(w)[0] for w in end])
            poss = [(p, q) for p, q in poss if helper.remove_punc(p.split()[-1]) in pos]

        poss = [(p, q) for p, q in poss if used_templates.count(p) < 2]

        if len(poss) == 0: return None, None
        t = self.fix_template(random.choice(poss))
        # t = self.fix_template(t[0]), t[1]
        if "<" in t[0]: t = (t[0].split("<")[0] + random.choice(t[0].split("<")[-1].strip(">").split("/")), t[1])
        return t[0], t[1]

    def fix_template(self, template):
        if type(template) == tuple: return self.fix_template(template[0]), template[1]
        if "he" in self.gender or "she" in self.gender:
            template = template.replace(" VBP", " VBZ").replace(" DO ", " DOES ")
        else:
            template = template.replace(" VBZ", " VBP").replace(" DOES ", " DO ")

        template = template.replace("UVBZ", "VBZ")

        if "VBC" in template and "PRPS" in template.split("VBC")[0]:  # if PRPS comes before PRPS
            if "he" in self.gender or "she" in self.gender:
                self.pos_to_words["VBC"] = {x: 1 for x in ["does", "seems", "appears", "looks"] + ["was"]}

            elif "i" in self.gender:
                self.pos_to_words["VBC"] = {x: 1 for x in ["do", "seem", "appear", "look"] + ["was"]}

            else:
                self.pos_to_words["VBC"] = {x: 1 for x in ["do", "seem", "appear", "look"] + ["were"]}

        if "<IS/AM>" in template:
            if "i" in self.gender:
                template = template.replace("<IS/AM>", "AM")

            elif "he" in self.gender or "she" in self.gender:
                template = template.replace("<IS/AM>", "IS")

            else:
                template = template.replace("<IS/AM>", "ARE")

        return template

    def get_poss_meters(self, template,
                        meter):  # template is a list of needed POS, meter is a string of the form "0101010..." or whatever meter remains to be assinged (but backward)
        """

        :param template: a list of POS's for the desired template
        :param meter: The desired meter for the line as a whole. Should be given backwords, i.e. "1010101010"
        :return: A dictionary with meters as keys mapping possible meter values for the last word in template to dicts in which the keys are the possible values
        the next word can take on, given that meter assigned to the last word.
        """
        if type(template) == str: template = template.split()

        word_pos = template[-1]
        if word_pos[-1] == ">":
            word_pos = word_pos.split("<")[0]
        elif word_pos[-1] in [",", ".", ":", ";", ">", "?"]:
            word_pos = word_pos[:-1]

        if word_pos == "POS":
            temp = self.get_poss_meters(template[:-1], meter)
            if temp != None:
                return {"": temp}

        if len(template) == 1:
            check_meter = "".join(reversed(meter))
            if (check_meter, word_pos) not in self.meter_and_pos or len(
                    self.meter_and_pos[(check_meter, word_pos)]) == 0:
                return None
            else:
                return {
                    check_meter: {}}  # should return a list of meters for the next word to take. in base case there is no next word, so dict is empty
        else:
            poss_meters = {}
            for poss_meter in self.possible_meters:
                # print("checking for ", word_pos, "with meter ", poss_meter)
                check_meter = "".join(
                    reversed(poss_meter))  # we're going through the string backwords, so we reverse it
                if meter.find(poss_meter) == 0 and (check_meter, word_pos) in self.meter_and_pos and len(
                        self.meter_and_pos[(check_meter, word_pos)]) > 0:
                    temp = self.get_poss_meters(template[:-1], meter[len(poss_meter):])
                    # print("made recursive call")
                    if temp != None:
                        # print("adding something to dict")
                        poss_meters[check_meter] = temp
            if len(poss_meters) == 0:
                return None
            return poss_meters

    def get_poss_meters_forward(self, template, meter,
                                rhyme_meters=None):  # template is a list of needed POS, meter is a string of the form "0101010..." or whatever meter remains to be assinged
        """
        :param template: A list of POS's and/or special words
        :param meter: The desired meter for the line, given forward as a string of the form "0101010101".
        :param rhyme_meters: a list of the meters of possible rhyming words
        :return: A dictionary with meters as keys mapping possible meter values for the last word in template to dicts in which the keys are the possible values
        the next word can take on, given that meter assigned to the last word.
        """
        if type(template) == str: template = template.split()

        word_pos = template[0]
        if word_pos[-1] == ">":
            word_pos = word_pos.split("<")[0]
        elif word_pos[-1] in [",", ".", ":", ";", ">", "?"]:
            word_pos = word_pos[:-1]

        if word_pos == "POS":
            temp = self.get_poss_meters_forward(template[1:], meter, rhyme_meters)
            if temp != None:
                return {"": self.get_poss_meters_forward(template[1:], meter, rhyme_meters)}

        if len(template) == 1:
            if (meter, word_pos) not in self.meter_and_pos or len(self.meter_and_pos[(meter, word_pos)]) == 0:
                return None
            elif rhyme_meters and meter not in rhyme_meters:
                return None
            else:
                return {
                    meter: {}}  # should return a list of meters for the next word to take. in base case there is no next word, so dict is empty
        else:
            poss_meters = {}
            for poss_meter in self.possible_meters:
                # print("checking for ", word_pos, "with meter ", poss_meter)
                if meter.find(poss_meter) == 0 and (poss_meter, word_pos) in self.meter_and_pos and len(
                        self.meter_and_pos[(poss_meter, word_pos)]) > 0:
                    temp = self.get_poss_meters_forward(template[1:], meter[len(poss_meter):], rhyme_meters)
                    # print("made recursive call")
                    if temp != None:
                        # print("adding something to dict")
                        poss_meters[poss_meter] = temp
            if len(poss_meters) == 0:
                return None
            return poss_meters

    def check_template(self, template,
                       meter,
                       verbose=True):  # makes sure any special words are in the meter dictionary, takes (template and meter as lists)
        """
        :param template: takes a template
        :param meter: takes the meter paired with the template, with words separated by _'s
        :return: returns nothing, just updates the dictionary
        """

        if type(template) == str: template = template.split()
        if type(meter) == str: meter = meter.split("_")

        for i in range(len(template)):
            word = template[i]
            if word[-1] == ">":
                word = word.split("<")[0]
            elif word[-1] in [",", ".", ":", ";", ">", "?"]:
                word = word[:-1]

            if word == "POS":
                continue
            if word in self.special_words:
                if (meter[i], word) not in self.meter_and_pos:
                    if verbose: print("adding to dictionary the word, ", word, "with meter ", meter[i])
                    self.meter_and_pos[(meter[i], word)] = [word]
                    self.dict_meters[word.lower()].append(meter[i])
                else:
                    self.meter_and_pos[(meter[i], word)].append([word])

    def write_line_dynamic_meter(self, template=None, meter=None, rhyme_word=None, n=1, verbose=False):
        if template is None: template, meter = random.choice(self.templates)
        template = self.fix_template(template)

        self.check_template(template, meter)

        if n > 1: return [self.write_line_dynamic_meter(template, meter, rhyme_word) for i in range(n)]

        print("writing line", template, " with dynamic meter")
        if rhyme_word and type(rhyme_word) == list: rhyme_word = rhyme_word[-1]
        if rhyme_word and verbose: print("rhyme word:", rhyme_word)
        if type(template) == str: template = template.split()

        line = ""
        punc = ",.;?"
        meters = []
        # my_meter_dict = self.get_poss_meters(template, "1010101010")
        my_meter_dict = self.get_poss_meters_forward(template, "01" * 5)
        if my_meter_dict == None:
            print("uh oh, this template can't work")
            return
        for i in range(len(template)):
            my_meter = random.choice(list(my_meter_dict.keys()))
            next_word = self.weighted_choice(template[i], my_meter)
            if not next_word: input("no word for " + template[i] + my_meter)
            space = " " * int(line != "" and next_word not in (punc + "'s"))
            line += space + next_word
            meters.append(my_meter)
            my_meter_dict = my_meter_dict[my_meter]

        new_word = ""
        while rhyme_word and not self.rhymes(new_word, rhyme_word):
            if verbose: print("trying to rhyme", template[-1], meters[-1], new_word, "with", rhyme_word)
            old_word = line.split()[-1].translate(str.maketrans('', '', string.punctuation))
            self.reset_letter_words()
            new_word = self.weighted_choice(template[-1], meters[-1], rhyme=rhyme_word).translate(
                str.maketrans('', '', string.punctuation))
            if verbose: print("got", new_word)
            if not new_word:
                print("cant rhyme")
                return 1 / 0
            line = line.replace(old_word, new_word)  # will replace all instances

        return line.strip()

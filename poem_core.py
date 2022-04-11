import helper

import random
import pickle
import numpy as np

import string
import pronouncing

from os import path



class Poem:
    def __init__(self, words_file="saved_objects/tagged_words.p",
                 templates_file=('poems/templates_basic.txt', "poems/rhetorical_templates.txt"),
                 syllables_file='saved_objects/cmudict-0.7b.txt',
                 extra_stress_file='saved_objects/edwins_extra_stresses.txt',
                 top_file='saved_objects/words/top_words.txt',
                 mistakes_file=None, tense=None):

        while mistakes_file and not path.exists(mistakes_file): mistakes_file = input(
            mistakes_file + "does not exist on your laptop, please enter your path now and/or when creating a poem object or change the code (ask edwin): ")
        keep_scores = False  # "byron" in words_file
        self.pos_to_words, self.words_to_pos = helper.get_new_pos_dict(words_file, mistakes_file=mistakes_file,
                                                                       keep_scores=keep_scores)
        self.backup_words = None
        if "byron" in words_file:
            self.backup_words, _ = helper.get_new_pos_dict('saved_objects/tagged_words.p')
            for pos in self.backup_words:
                if pos not in self.pos_to_words:
                    self.pos_to_words[pos] = self.backup_words[pos]
                    for word in self.backup_words[pos]:
                        if word not in self.words_to_pos:
                            self.words_to_pos[word] = []
                        self.words_to_pos[word].append(pos)

        if type(templates_file) == str: templates_file = [templates_file]
        self.templates = []
        for t in templates_file:
            try:
                #with open(t) as tf:
                #    self.templates += [(" ".join(line.split()[:-1]), line.split()[-1]) for line in tf.readlines() if
                #                       "#" not in line and len(line) > 1]
                self.templates += self.get_templates_from_file(t)
            except:
                print(t, " does not exist so reading from poems/templates_basic.txt instead")
                with open("poems/templates_basic.txt") as tf:
                    self.templates = [(" ".join(line.split()[:-1]), line.split()[-1]) for line in tf.readlines() if
                                      "#" not in line and len(line) > 1]

        self.all_templates_dict = {}
        for file in ["templates_present", "templates_past", "templates_future", "templates_basic"]:
            file_path = "poems/" + file + ".txt"
            f = open(file_path, "r")
            for line in f.readlines():
                self.all_templates_dict[" ".join(line.split()[:-1])] = line.split()[-1]

        self.special_words = helper.get_finer_pos_words()

        self.dict_meters = helper.create_syll_dict([syllables_file], extra_stress_file)

        self.pron = {w.split()[0].lower(): " ".join(w.split()[1:]) for w in open(syllables_file).readlines() if
                     w.split()[0].lower().split("(")[0] in self.words_to_pos}

        with open(top_file) as tf:
            self.top_common_words = [line.strip() for line in tf.readlines()]#[:125]

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

        self.tense = tense

        self.save_poems = False

        self.prev_rhyme = None

    def get_meter(self, word):
        word = word.strip().replace(" ", "_")
        if not word or len(word) == 0: return [""]
        if word[-1] in ".,?;":
            return self.get_meter(word[:-1])
        elif word[-1] == ">":
            return self.get_meter(word.split("<")[0])

        if word not in self.dict_meters: return []

        if any(len(j) == 1 for j in self.dict_meters[word]): return list(set(["0", "1"] + self.dict_meters[word]))
        return self.dict_meters[word]

    def get_word_pos(self, word):
        """
        Get the set of POS category of a word. If we are unable to get the category, return None.
        """
        # Special case
        word = helper.remove_punc(word).strip().replace(" ", "_")
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
        if meter and type(meter) == str:
            meter = [meter]

        if "PRP" in pos and "_" not in pos:
            if pos == "PRPOO":
                #print("here", self.pos_to_words['PRPO'], self.gender)
                ret = [p for p in self.pos_to_words["PRPO"] if p not in self.gender] #PRPOO gets different pronouns
            else:
                ret = [p for p in self.pos_to_words[pos] if p in self.gender]
            if meter:
                ret = [r for r in ret if any(q in meter for q in self.get_meter(r))]
            return ret
        elif pos not in self.pos_to_words:
            return []

        if pos not in self.pos_to_words:
            return self.get_backup_pos_words(pos=pos, meter=meter, rhyme=rhyme)

        if meter:
            ret = [word.replace("_", " ") for word in self.pos_to_words[pos] if any(m in self.get_meter(word) for m in meter)]
            return ret
        return [p.replace("_", " ") for p in self.pos_to_words[pos]]

    def get_backup_pos_words(self, pos, meter=None, rhyme=None):
        if not self.backup_words: return []
        temp = self.pos_to_words
        self.pos_to_words = self.backup_words
        words = self.get_pos_words(pos, meter, rhyme)
        self.pos_to_words = temp
        return words

    def rhymes(self, word1, word2, check_cmu=False):
        if not word1 or not word2: return False
        if word1[-1] in ".,?!>": word1 = word1.translate(str.maketrans('', '', string.punctuation))
        if word2[-1] in ".,?!>": word2 = word2.translate(str.maketrans('', '', string.punctuation))
        if word1 == word2: return False
        if self.prev_rhyme is not None and word1 in self.get_rhyme_words(self.prev_rhyme): return False
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
        return [w for w in pronouncing.rhymes(word) if w in self.words_to_pos]

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

    def get_rhymes(self, theme, words):
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
            t in self.end_pos[pos] for t in self.get_meter(word) for pos in self.get_word_pos(word) if
            pos in self.end_pos)

    def write_line_gpt(self, template=None, meter={}, rhyme_word=None, n=1, gpt_model=None, flex_meter=True,
                       all_verbs=False, verbose=False, alliteration=None, internal_rhymes=[], theme_words=[], theme_threshold=0.5):
        if not self.gpt:
            # self.gpt = gpt_2_gen.gpt(seed=None, sonnet_method=self.get_pos_words)
            self.gpt = gpt_model
            if not gpt_model: print("need a gpt model", 1 / 0)

        if n > 1: return [self.write_line_gpt(template, meter, rhyme_word, flex_meter=flex_meter, all_verbs=all_verbs, internal_rhymes=internal_rhymes,
                                              verbose=verbose, alliteration=alliteration, theme_words=theme_words, theme_threshold=theme_threshold) for _ in range(n)]

        if not meter: flex_meter = False
        if template is None: template, meter = random.choice(self.templates)

        #template = self.fix_template(template)
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
                    if (meter and len(r) == 0) or len(rhyme_words) == 0:
                        if verbose: print("couldn't get a rhyme here:", template, rhyme_word, rhyme_words, r)
                        return None
            else:
                r = None

            if "_" in base_template:
                meter_dict = self.get_poss_meters_forward_rhet(base_template, "".join(meter.split("_")), {}, r)

            else:
                meter_dict = self.get_poss_meters_forward(base_template, "01" * 5, r)
            if not meter_dict:
                if verbose: print("couldn't get a meter_dict:", template, rhyme_word)
                return None
            if verbose: print("writing flexible line", template, meter_dict, rhyme_word)

            return self.gpt.generation_flex_meter(template.split(), meter_dict, seed=self.gpt_past,
                                                  rhyme_word=rhyme_word, verbose=verbose, alliteration=alliteration, internal_rhymes=internal_rhymes,
                                                  theme_words=theme_words, theme_threshold=theme_threshold)

        else:
            if verbose: print("writing line", template, meter)
            # if n > 1: return [self.gpt.good_generation(template=template.split(), meter=meter.split("_"), rhyme_word=rhyme_word, verbose=verbose) for i in range(n)]

            rhyme_pos = helper.remove_punc(template.split()[-1]).split("_")[-1]
            rhyme_words = self.get_pos_words(rhyme_pos, rhyme=rhyme_word) if type(rhyme_word) == str else rhyme_word

            if rhyme_words is not None and len(rhyme_words) == 0:
                if verbose: print("couldn't get a rhyme here:", template, rhyme_word, rhyme_words)
                return None
            print("my rhymes are", rhyme_words)
            print("the rhyme pos i found was", rhyme_pos)
            print("this is for the rhyme word", rhyme_word)
            return self.gpt.generation_flex_meter(template.split(), meter_dict={}, seed=self.gpt_past, internal_rhymes=internal_rhymes,
                                                  rhyme_word=rhyme_word, verbose=verbose, alliteration=alliteration, theme_words=theme_words, theme_threshold=theme_threshold)


    def get_meter_dict(self, template, meter, rhyme_word=None, verbose=False):

        base_template = template.replace("*VB", "VB").replace("sc", "")
        self.check_template(base_template, meter)
        rhyme_pos = helper.remove_punc(template.split()[-1]).split("_")[-1]
        # if "_" in rhyme_pos
        if rhyme_word:

            rhyme_words = self.get_pos_words(rhyme_pos, rhyme=rhyme_word) if type(rhyme_word) == str else rhyme_word

            r = set([x for x in ["1", "01", "101", "0101", "10101"] for w in rhyme_words if
                     x in self.get_meter(w)]) if rhyme_word else None
            if (meter and len(r) == 0) or len(rhyme_words) == 0:
                if verbose: print("couldn't get a rhyme here:", template, rhyme_pos, rhyme_word, rhyme_words, r)
                return None
        else:
            r = None

        if "_" in base_template:
            meter_dict = self.get_poss_meters_forward_rhet(base_template, "".join(meter.split("_")), {}, r)

        else:
            meter_dict = self.get_poss_meters_forward(base_template, "01" * 5, r)
        if not meter_dict:
            if verbose: print("couldn't get a meter_dict:", template, rhyme_word)
            return None

        return meter_dict

    def write_line_random(self, template=None, meter=None, rhyme_word=None, n=1, verbose=False):
        if template is None: template, meter = random.choice([t for t in self.templates if "_" not in t[0]])
        #template = self.fix_template(template)

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

    def reset_gender(self, choice=None):
        pronouns = [["i", "me", "my", "mine", "myself"], ["you", "your", "yours", "yourself"], ["he", "him", "his", "himself"],
             ["she", "her", "hers", "herself"], ["we", "us", "our", "ours", "ourselves"],
             ["they", "them", "their", "theirs", "themselves"]]

        if choice is None:
            self.gender = random.choice(pronouns)
        elif choice == "all":
            #self.gender = [p for p in sub_list for sub_list in pronouns]
            self.gender = [p for sub_list in pronouns for p in sub_list]
        else:
            raise ValueError("not implemented")

        g = random.choice(["male", "female"])
        if "he" in self.gender:
            g = "male"
        elif "she" in self.gender:
            g = "female"

        #self.pos_to_words["NAM"] = {n: 1 for n in self.all_names[g]}

        self.reset_meter_pos_dict()

    def set_meter_pos_dict(self):
        self.meter_and_pos = {}
        possible_pos = list(self.pos_to_words.keys())
        for pos in possible_pos:
            for meter in self.possible_meters:
                if "PRP" in pos:
                    self.meter_and_pos[(meter, pos)] = [word for word in self.pos_to_words[pos] if
                                                        word in self.dict_meters and meter in self.get_meter(word)
                                                        and word in self.gender]
                else:
                    self.meter_and_pos[(meter, pos)] = [word for word in self.pos_to_words[pos] if
                                                        word in self.dict_meters and meter in self.get_meter(word)]
        for word in self.special_words:
            if word in self.dict_meters:
                meter = self.dict_meters[word]
                self.meter_and_pos[(meter, word)] = [word]
            else:
                continue

        for meter,pos in list(self.meter_and_pos.keys()):
            if meter == "1":
                self.meter_and_pos[("0", pos)] += self.meter_and_pos[(meter,pos)]
                self.meter_and_pos[("0", pos)] = list(set(self.meter_and_pos[("0", pos)]))
            if meter == "0":
                self.meter_and_pos[("1", pos)] += self.meter_and_pos[(meter,pos)]
                self.meter_and_pos[("1", pos)] = list(set(self.meter_and_pos[("1", pos)]))

    def reset_meter_pos_dict(self):
        possible_pos = [item for item in list(self.pos_to_words.keys()) if "PRP" in item]
        for pos in possible_pos:
            for meter in self.possible_meters:
                self.meter_and_pos[(meter, pos)] = [word for word in self.pos_to_words[pos] if
                                                    word in self.dict_meters and meter in self.get_meter(word) and word in self.gender]

    def get_templates_from_file(self, filename):
        with open(filename) as tf:
            templates = [(" ".join(line.split()[:-1]), line.split()[-1]) for line in tf.readlines() if
                              "#" not in line and len(line) > 1]

        return templates

    def get_template_from_line(self, line, backwards=False):
        self.reset_gender("all")

        words = line if type(line) == list else line.lower().split()
        poss = list(self.all_templates_dict.keys())
        if not backwards:
            for i, word in enumerate(words):
                word = word.split("_")[0]
                poss = [p for p in poss if helper.remove_punc(p.split()[i]) in self.get_word_pos(word)]
                if len(poss) == 1: return poss
        else:
            for i in range(-1, -len(words) - 1, -1):
                poss = [p for p in poss if helper.remove_punc(p.split()[i]) in self.get_word_pos(words[i])]
                if len(poss) == 1: return poss
        return poss

    def get_next_template(self, used_templates, end=""):
        """

        Parameters
        ----------
        used_templates
        check_the_rhyme
        end - makes sure the template could have that word at the end (but only if it starts with __)

        Returns
        -------

        """
        if type(used_templates) == str:
            used_templates = [used_templates]

        if len(used_templates) > 0 and type(used_templates[0]) == tuple: used_templates = [u[0] for u in used_templates]
        poss = [(p, q) for (p, q) in self.templates if used_templates.count(p) < 2 and (not used_templates or p != used_templates[-1])]
        # L- makes sure that a template isn't used more than twice
        # L- second part of and statement makes sure that the next possible template is not the same as the most recently
        # used template
        # incomplete = ",;" + string.ascii_lowercase
        n = len(used_templates)
        if n > 0:
            gerund_templates = ["RB VBG VBD WHERE ALL PRPD$ JJ NNS VB,", "FOR VBG NN WITH PRPO RB,",
                              "VBG A NN WHERE NNS VB,", "AND VBG A NN WHERE NNS VB,", "AND VBG PRPD$ NN BY PRPD$ NN,"]
            #gerund_templates = []
            if used_templates[-1] == "FOR JJS NNS, PRPS VBP NNS": #checked 4/2/21
                poss = [("TO VB WITHIN PRPD$ JJ JJ JJ NNS,", "0_1_01_0_1_0_10_1"),
                        ("TO VB THE NN POS NN BY THE NN AND VB", "0_1_0_1__0_1_0_1_0_1"),
                        ("TO VB THE NNS TO THE JJ NN.", "0_1_0_10_1_0_10_1"),
                        ("BUT THE NN SHOULD BY NN VB", "0_1_010_1_0_1_01")]
            elif used_templates[-1] == "BUT PRPS VBD TO PRPD$ JJ NNS": #checked 4/2/21
                poss = [("SO PRPS THROUGH NNS OF JJ NNS VBD.", "0_1_0_10_1_0_1_01"),
                        ("SO TOO PRPD$ NNS VBD TO PRPD$ NN.", "0_1_0_10_1_0_1_01"),
                        ("AND ABNN WAS A JJ AND JJ NN.", "0_10_1_0_1_0_1_01"),
                        ("AND VBD PRPD$ NN BY PRPD$ NN<,/.>", "0_10_1_01_0_1_01"),
                        ("SO JJ A NN OF NNS, YET PRPS VBD RB", "0_1_0_1_0_1_0_1_0_1")]
            elif helper.remove_punc(used_templates[-1]) in "IF PRPS COULD VB THIS JJ NN OF ABNN<./,>": #checked 4/2/21
                poss = [("THERE WILL BE, PRPD$ JJ NN, MUCH OF;", "0_1_0_1_01_01_0_1"),
                        ("THE JJ NN WILL BE PRPD$ JJS ABNN.", "0_10_1_0_1_0_10_1"),
                        ("THE ABNN, LIKE NNS IN THE NN", "0_1_0_101_0_1_01"),
                        ("SO TOO PRPD$ NNS WILL VB PRPD$ NN<;/.>", "0_1_0_10_1_01_0_1"),
                        ("SO PRPD$ NNS WILL VB PRPD$ NN<;/.>", "0_1_01_0_10_0_1")]
            elif used_templates[-1] == "BUT IF PRPS VBP PRPO TO THE NN,": #checked 4/2/21
                poss = [("THERE WILL BE, PRPD$ JJ NN, MUCH OF;", "0_1_0_1_01_01_0_1"),
                        ("THE JJ NN WILL BE PRPD$ JJS ABNN.", "0_10_1_0_1_0_10_1"),
                        ("THE ABNN, LIKE NNS IN THE NN", "0_1_0_101_0_1_01"),
                        ("SO TOO PRPD$ NNS WILL VB TO PRPD$ NN<;/.>", "0_1_0_10_1_0_1_0_1")]
            elif used_templates[-1] == "WHEN PRPS VBD THE JJ, VBD NN": #checked 4/2/21
                poss = [("SO TOO PRPD$ NNS VBD TO PRPD$ NN<;/.>", "0_1_0_1_01_0_1_01"),
                        ("THE NN VBBD LIKE A NN IN THE NN<./,/;>", "0_1_01_0_1_0_1_0_1"),
                        ("ABNN VBD AND VBD EVERY WHERE<,/.>", "01_01_0_10_10_1"),
                        ("PRPD$ JJ NN COULD VB RB WITH PRPO.", "0_10_10_1_0_1_0_1"),
                        ("PRPS VBD AN ALL JJ NN AND JJ NN.", "0_1_0_1_0_1_0_10_1")]
            elif used_templates[-1] == "FROM JJS NNS, PRPS VBP RB": #checked 4/2/21
                poss = [("TO VB WITHIN PRPD$ JJ JJ JJ NNS,", "0_1_01_0_1_0_10_1"),
                        ("TO VB THE NN POS NN BY THE NN AND VB", "0_1_0_1__0_1_0_1_0_1"),
                        ("TO VB THE NNS TO THE JJ NN.", "0_1_0_10_1_0_10_1"),
                        ("BUT AS THE NN SHOULD BY NN VB,", "0_1_0_10_1_0_1_01"),
                        ("A JJ NN VBD IN NNS OF NNS<,/.>", "0_10_10_1_0_1_0_1")]
            elif used_templates[-1] == "AS THE JJ NN OF THE NNS":
                poss = [("VB RB ON JJ AND JJ NNS", "0_10_0_1_0_10_10"),
                        ("VBZ RB ON JJ AND JJ NNS", "0_10_0_1_0_10_10"),
                        ("VB RB ON JJ AND JJ NN", "0_10_0_1_0_10_10")]

            elif used_templates[-1] in gerund_templates:
                followers = [("PRPS VBZ TO THOSE THAT VB RB", "0_1_0_1_0_10_101"),
                        ("PRPS VBZ AND VBZ TO THOSE THAT VB RB", "0_1_0_10_1_0_1_0_1"),
                        ("PRPS VBZ WHERE THE NNS VB", "0_10_1_0_10_101"),
                        ("PRPS VBZ TO THE NN WHERE PRPD$ NNS VB", "0_1_0_1_01_0_1_0_1")]
                poss = list(set(followers).intersection(set(poss)))


            if used_templates[-1][-1] in ".?":
                poss = [p for p in poss if p[0].split()[0] not in ["AND", "THAT", "OR", "SHALL", "WILL", "WHOSE", "TO", "WAS", "VBD", "IN"]]

            # L- ensure you can't have a stanza of 4 separate sentences
            # ensure that if the first two lines in a stanza ended in a period, then the next one shouldn't
            # (though maybe first one or first three?)
            # Beware of templates that end like "<./,>" - you're allowed to replace that with just "," if you want
            if n % 4 == 2 and (used_templates[-1][-1] == "." and used_templates[-2][-1] == "."):
                poss = [(p, q) for p, q in poss if p[-1] != '.']
                poss = [(p.replace("/.", "").replace("<./", "<"), q) for p, q in poss if p[-1] in ".>"]
            # L- for the second line in every stanza, if the two most recently used templates both ended in '.',
            # remove next possible templates that end in '.'

            if n % 4 == 3 or n == 13:
                # poss = [p for p in poss if p[0][-1] not in ",;" + string.ascii_uppercase]
                poss = [(p.replace("/,", "").replace("<,/", "<"), q) for p, q in poss if p[-1] in ">."]
                # L- for the third line in every stanza or the second-to-last line in the sonnet, remove comma as punctuation
                # choice for next possible template
                # print("last line of stanza so:", poss)

            if n % 4 == 0:
                poss = [(p, q) for p, q in poss if p.split()[0] not in ["AND", "OR"]]
            elif sum([int("_" in t) for t in used_templates]) > 1:
                poss = [(p,q) for p,q in poss if "_" not in p]
            # L- for the last line in every stanza, the next possible template cannot begin with 'and' or 'or'

            if n % 4 > 1 or n == 13:
                poss = [(p, q) for p, q in poss if "_" not in p.split()[-1]]
            #for the second and third line in every stanza or the second-to-last line in the sonnet, drop next possible
            # L- templates that end in '_' (are there any basic templates like this?)

        else:
            # starting templates taken from google doc
            starters = [("A JJ NN VBD IN NNS OF NN<,/.>", "0_10_10_1_0_1_0_1"),
                    ("WHAT JJ NN VBZ PRPD$ NN?", "0_1010_10_1_0_1"),
                    ("PRPS VBC JJ TO VB THE NNS", "0_1_01_0_1_0_101"),
                    ("FROM JJS NNS, PRPS VBP RB", "0_10_10_1_01_01"),
                    ("THE NN OF NN ON A JJ N", "0_1_0_10_1_0_10_1"),
                    ("RB VBG LIKE A NN VBG", "010_10_1_0_10_1"),
                    ("A JJ NN FROM THE JJ NN", "0_10_10_1_0_1_01"),
                    ("WHEN ALL THE NNS OF THIS NN ARE JJ,", "0_1_0_10_1_0_1_0_1"),
                    ("WHY VBC PRPS VBP SUCH A JJ NN?", "0_1_0_10_1_0_101_0"),
                    ("AS THE JJ NN OF THE NNS", "0_1_01_01_0_1_01"),
                    ("FOR ALL THE ABNN THAT DOES VB PRPO,", "0_1_0_10_1_0_1_0_1"),
                    ("IN ABNN PRPS DO NOT VB PRPOO WITH PRPD$ NN,", "0_1_0_1_0_1_0_1_0_1")]

            poss = [p for p in starters if p in self.templates]

            if len(poss) == 0: poss = starters

        if len(poss) == 0:
            print("there are no templates ", str(len(used_templates)), used_templates[-1], end)
            return 1/0
            #return self.fix_template(random.choice(self.templates))

        #if check_the_rhyme: poss = [p for p in poss if any(
         #   self.rhymes(check_the_rhyme, w) for w in self.get_pos_words(p[0].split()[-1], p[1].split("_")[-1]))]
        # poss = [(p, q) for p, q in poss if used_templates.count(p) < 2]
        # L- if a template has been used 2 or more times in the sonnet already, it cannot be used again

        if len(poss) == 0: return None, None
        t = self.fix_template(random.choice(poss))
        if t not in self.templates:
            self.templates.append(t)
        # t = self.fix_template(t[0]), t[1]
        if "<" in t[0]: t = (t[0].split("<")[0] + random.choice(t[0].split("<")[-1].strip(">").split("/")), t[1])

        return t[0], t[1]
        # L- chooses punctuation between <>

    def fix_template(self, template):
        if type(template) == tuple: return self.fix_template(template[0]), template[1]
        if "he" in self.gender or "she" in self.gender:
            template = template.replace(" VBP", " VBZ").replace(" DO ", " DOES ")
        else:
            template = template.replace(" VBZ", " VBP").replace(" DOES ", " DO ")

        template = template.replace("UVBZ", "VBZ").replace("UVBP", "VBP")

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
                       verbose=False):  # makes sure any special words are in the meter dictionary, takes (template and meter as lists)
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
        #template = self.fix_template(template)

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

    def get_poss_meters_no_template(self, desired_meter="0101010101"):

        if isinstance(desired_meter, list):
            desired_meter = "".join(desired_meter)
        meter_options = set([key[0] for key in self.meter_and_pos.keys() if len(self.meter_and_pos[key]) > 0])
        if desired_meter == "":
            return {}
        else:
            poss_meters = {}
            for poss_meter in self.possible_meters:
                # print("checking for ", word_pos, "with meter ", poss_meter)
                if desired_meter.find(poss_meter) == 0 and poss_meter in meter_options:
                    temp = self.get_poss_meters_no_template(desired_meter[len(poss_meter):])
                    # print("made recursive call")
                    if temp != None:
                        # print("adding something to dict")
                        poss_meters[poss_meter] = temp
            if len(poss_meters) == 0:
                return None
            return poss_meters

    def get_poss_meters_forward_rhet(self, template, meter, rhet_dict,
                                rhyme_meters=None):  # template is a list of needed POS, meter is a string of the form "0101010..." or whatever meter remains to be assinged
        """
        :param template: A list of POS's and/or special words
        :param meter: The desired meter for the line, given forward as a string of the form "0101010101".
        :param rhyme_meters: a list of the meters of possible rhyming words
        :return: A dictionary with meters as keys mapping possible meter values for the last word in template to dicts in which the keys are the possible values
        the next word can take on, given that meter assigned to the last word.
        """
        if type(template) == str: template = template.split()

        pair_info = None

        word_pos = template[0]
        if word_pos[-1] == ">":
            word_pos = word_pos.split("<")[0]
        elif word_pos[-1] in [",", ".", ":", ";", ">", "?"]:
            word_pos = word_pos[:-1]
        if "_" in word_pos:
            rhet_info = word_pos.split("_")
            word_pos = rhet_info[0]
            pair_info = rhet_info[1]

        if word_pos == "POS":
            temp = self.get_poss_meters_forward_rhet(template[1:], meter, rhet_dict, rhyme_meters)
            if temp != None:
                return {"": self.get_poss_meters_forward_rhet(template[1:], meter, rhet_dict, rhyme_meters)}

        if len(template) == 1:
            if (pair_info != None and meter != rhet_dict[helper.remove_punc(pair_info)]) or (meter, word_pos) not in self.meter_and_pos or len(self.meter_and_pos[(meter, word_pos)]) == 0:
                return None
            elif rhyme_meters and meter not in rhyme_meters:
                return None
            else:
                return {meter: {}}  # should return a list of meters for the next word to take. in base case there is no next word, so dict is empty
        else:
            poss_meters = {}
            for poss_meter in self.possible_meters:
                # print("checking for ", word_pos, "with meter ", poss_meter)
                if meter.find(poss_meter) == 0 and (poss_meter, word_pos) in self.meter_and_pos and len(
                        self.meter_and_pos[(poss_meter, word_pos)]) > 0:
                    if pair_info != None and pair_info in rhet_dict and rhet_dict[pair_info] != poss_meter:
                        continue
                    if pair_info != None:
                        temp_dict = rhet_dict.copy()
                        temp_dict[pair_info] = poss_meter
                        temp = self.get_poss_meters_forward_rhet(template[1:], meter[len(poss_meter):], temp_dict, rhyme_meters)
                    else:
                        temp = self.get_poss_meters_forward_rhet(template[1:], meter[len(poss_meter):], rhet_dict, rhyme_meters)
                    # print("made recursive call")
                    if temp != None:
                        # print("adding something to dict")
                        poss_meters[poss_meter] = temp
            if len(poss_meters) == 0:
                return None
            return poss_meters

    def write_line_gpt_no_template(self, meter={}, rhyme_word=None, n=1, gpt_model=None, flex_meter=True, verbose=False, alliteration=None, theme_words=[], theme_threshold=0.5):
        if not self.gpt:
            # self.gpt = gpt_2_gen.gpt(seed=None, sonnet_method=self.get_pos_words)
            self.gpt = gpt_model
            if not gpt_model: print("need a gpt model", 1 / 0)
        #TODO figure out seed stuff
        if not self.gpt_past or self.gpt_past.strip() == "":
            self.gpt_past = "Shall I compare thee to a summer's day? Thou art more lovely and more temperate."
        print("GPT past: ", self.gpt_past)
        return self.gpt.gen_line_no_template(seed=self.gpt_past)



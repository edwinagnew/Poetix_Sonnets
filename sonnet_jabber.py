import pickle
import helper
import random
import line



class Sonnet_Gen():
    def __init__(self,postag_file='saved_objects/postag_dict_all+VBN.p',
                 syllables_files=['saved_objects/cmudict-0.7b.txt', 'saved_objects/ob_syll_dict.txt'],
                 extra_stress_file='saved_objects/edwins_extra_stresses.txt'):


        with open("saved_objects/ob_postag_dict.p", "rb") as pick_in:
            ob_postag_dict = pickle.load(pick_in)
            self.pos_to_words = ob_postag_dict[0]
            self.words_to_pos = ob_postag_dict[1]

        with open(postag_file, 'rb') as f:
            postag_dict = pickle.load(f)
        for pos in postag_dict[1]:
            if pos not in self.pos_to_words:
                self.pos_to_words[pos] = postag_dict[1][pos]
        for word in postag_dict[2]:
            if any(po not in self.pos_to_words for po in postag_dict[2][word]):
                print(word, "hi")
                print(1/0)
                self.words_to_pos[word] = postag_dict[2][word]

        self.special_words = helper.get_finer_pos_words()

        self.api_url = 'https://api.datamuse.com/words'

        self.dict_meters = helper.create_syll_dict(syllables_files, extra_stress_file)

        with open("poems/end_pos.txt", "r") as pickin:
            list = pickin.readlines()
            self.end_pos = {}
            for l in list:
                self.end_pos[l.split()[0]] = l.split()[1:]


        with open("poems/shakespeare_templates.txt", "r") as templs:
            self.templates = {}
            lines = templs.readlines()
            for line in lines:
                self.templates[" ".join(line.split()[:-1])] = line.split()[-1].strip()



    def gen_poem_jabber(self):

        rhyme_dict = {}
        for i in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            # rhyme_dict[i] = self.getRhymes([prompt], words=ob_pos_to_words['NN'])
            rhyme_dict[i] = self.get_ob_rhymes()
        last_word_dict = self.last_word_dict(rhyme_dict)

        candidates = ["         ----" + "jabber".upper() + "----"]
        used_templates = []
        for line_number in range(1, 15):
            first_word = random.choice(list(last_word_dict[line_number]))  # last word is decided
            while first_word not in self.dict_meters.keys() or not self.suitable_last_word(first_word):
                first_word = random.choice(list(last_word_dict[line_number]))
            in_template = self.get_word_pos(first_word)[0]
            while in_template not in self.end_pos or not any(pos in self.end_pos[in_template] for pos in self.dict_meters[first_word]):
                in_template = random.choice(self.get_word_pos(first_word))
            poss_meters = [poss_meter for poss_meter in self.dict_meters[first_word] if
                           poss_meter in self.end_pos[in_template]]
            if len(poss_meters) != 1: print(poss_meters, first_word, in_template)
            in_meter = poss_meters[0]
            curr_line = line.Line(first_word, self.dict_meters, pos_template=in_template)
            curr_line.meter = in_meter
            curr_line.syllables = len(in_meter)
            #curr_line.print_info()
            template = False
            reset = False
            while curr_line.syllables < 10:
                if reset:
                    print("HI", curr_line.text, curr_line.pos_template)
                    template = False
                if not template:
                    template = self.get_random_template(curr_line.pos_template, curr_line.meter)
                while not template:
                    print("no template", curr_line.pos_template, curr_line.text, curr_line.meter)
                    print(1/0)
                    first_w = curr_line.text.split()[0]
                    first_pos = self.get_word_pos(first_w)
                    if len(first_pos) > 1:
                        curr_line.pos_template = random.choice(first_pos) + curr_line.pos_template[
                                                                            len(curr_line.pos_template.split()[0]):]
                        template = self.get_random_template(curr_line.pos_template, curr_line.meter)
                    else:
                        print("unfixable")
                        print(1 / 0)

                if template in used_templates:
                    template = self.get_random_template(curr_line.pos_template, curr_line.meter)

                if template == curr_line.pos_template:
                    # NOT GREAT
                    print(curr_line.text, curr_line.pos_template, curr_line.meter)
                    print("here", 1/0)
                    curr_line.syllables = 100
                    continue
                next_pos = template.split()[-len(curr_line.pos_template.split()) - 1]
                next_meter = self.templates[template].split("_")[-len(curr_line.pos_template.split()) - 1]
                poss_words = self.get_pos_words(next_pos, meter=next_meter)
                reset = False
                while not poss_words:
                    if next_meter == "0":
                        print("0 fix", next_pos)
                        poss_words = self.get_pos_words(next_pos, meter="1") #cheeky fix
                    elif next_meter == "01" and next_pos == "VBG":
                        print("VBG fix")
                        poss_words = self.get_pos_words(next_pos, meter="10")
                    else:
                        print("template: ", template, "next_pos:", next_pos, "next_meter:", next_meter)
                        print("doesnt look like theres a ", next_pos, "with meter", next_meter)
                        # print("resetting", 1 / 0)
                        curr_line.reset()
                        reset = True
                        print("goodbye", curr_line.text)
                        break
                if not reset:
                    next_word = random.choice(poss_words)
                    curr_line.add_word(next_word, next_meter)
                    curr_line.pos_template = next_pos + " " + curr_line.pos_template
                    template = False  # make a parameter?

            print("adding line", line_number)
            curr_line.print_info()
            candidates.append(curr_line)
            used_templates.append(curr_line.pos_template)

        print("")
        print(candidates[0])
        del candidates[0]
        for cand in range(len(candidates)):
            print(candidates[cand].text)
            if ((cand + 1) % 4 == 0): print("")
        return candidates

    def get_ob_rhymes(self, syll_file="saved_objects/ob_syll_dict.txt", rhyme_file="saved_objects/saved_rhymes_jabb"):
        try:
            with open(rhyme_file, "rb") as pickle_in:
                rhymes = pickle.load(pickle_in)
        except:
            print("loading rhymes....")
            with open(syll_file, "r") as f:
                lines = [lin.split() for lin in f.readlines() if (";;;" not in lin)]

            rhymes = {}
            # go through all lines
            for l in lines:
                word = l[0].lower()
                if word in self.words_to_pos and self.suitable_last_word(word):
                    rhys = []
                    for l2 in lines:
                        if l[-2:] == l2[-2:] and l != l2: #for now if and only if last 2 syllables
                            rhys.append(l2[0].lower())
                    if len(rhys) > 0:
                        print(word, rhys)
                        rhymes[word] = rhys
            with open(rhyme_file, "wb") as pickle_in:
                pickle.dump(rhymes, pickle_in)
                print("loaded")
        return rhymes

    def suitable_last_word(self, word):  # checks pos is in self.end_pos and has correct possible meters
        return any(w in self.end_pos.keys() for w in self.get_word_pos(word)) and any(
            t in self.end_pos[pos] for t in self.dict_meters[word] for pos in self.get_word_pos(word) if
            pos in self.end_pos)

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

    def get_random_template(self, curr_template, curr_meter):
        poss_templates = [item for item in self.templates.keys() if item[-len(curr_template):] == curr_template and self.templates[item].split('_')[-len(curr_meter.split('_')):] == curr_meter.split('_')]
        if len(poss_templates) == 0: return False
        return random.choice(poss_templates)

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
        scheme = {1: 'A', 2: 'B', 3: 'A', 4: 'B', 5: 'C', 6: 'D', 7: 'C', 8: 'D', 9: 'E', 10: 'F', 11: 'E', 12: 'F',
                  13: 'G', 14: 'G'}
        last_word_dict = {}

        rand_keys = {}
        # for sc in ['A', 'B', 'C', 'D', 'E', 'F', 'G']: rand_keys[sc] = random.choice(list(rhyme_dict[sc].keys()))
        first_rhymes = []
        for i in range(1, 15):
            if i in [1, 2, 5, 6, 9, 10, 13]:  # lines with a new rhyme
                last_word_dict[i] = [random.choice(list(rhyme_dict[scheme[i]].keys()))]
                while len(last_word_dict[i]) < 1 or not self.suitable_last_word(last_word_dict[i][0]) or last_word_dict[i][0] in first_rhymes or any(
                        w in rhyme_dict['A'][word] for word in first_rhymes for w in rhyme_dict['A'][last_word_dict[i][0]]):
                    last_word_dict[i] = [random.choice(list(rhyme_dict[scheme[i]].keys()))]
                first_rhymes.append(last_word_dict[i][0])
            if i in [3, 4, 7, 8, 11, 12, 14]:  # lines with an old line
                # last_word_dict[i] = rhyme_dict[scheme[i]][rand_keys[scheme[i]]]
                letter = scheme[i]
                pair = last_word_dict[i - 2][0]
                if i == 14:
                    pair = last_word_dict[13][0]
                last_word_dict[i] = [word for word in rhyme_dict[letter][pair] if word in self.words_to_pos and self.suitable_last_word(word)]
                if len(last_word_dict[i]) < 1:
                    print("help: ", i, last_word_dict[i])
                    print(1/0)
        return last_word_dict








import numpy as np

import helper
import random
from py_files import line

import string


import poem_core


#Based off limericks.py

class Sonnet_Gen(poem_core.Poem):
    def __init__(self,words_file="saved_objects/tagged_words.p",
                 syllables_file='saved_objects/cmudict-0.7b.txt',
                 top_file='saved_objects/words/top_words.txt' ,
                 extra_stress_file='saved_objects/edwins_extra_stresses.txt',
                 templates_file = 'poems/jordan_templates.txt',
                 mistakes_file=None,
                 prompt=False):
        #self.pos_to_words, self.words_to_pos = helper.get_pos_dict(postag_file, mistakes_file=mistakes_file)
        poem_core.Poem.__init__(self, words_file=words_file, templates_file=templates_file,
                                syllables_file=syllables_file, extra_stress_file=extra_stress_file, top_file=top_file,
                                mistakes_file=mistakes_file)

        with open(templates_file, "r") as templs:
            self.templates = {}
            lines = templs.readlines()
            for lin in lines:
                self.templates[" ".join(lin.split()[:-1])] = lin.split()[-1].strip()

        self.end_pos = {}
        for temp in self.templates:
            end = temp.split()[-1]
            ends = [end]
            if "<" in end:
                ends = [end.split("<")[0] + end.split(">")[1].strip(">").split("/")[0], end.split("<")[0] + end.split(">")[1].strip(">").split("/")[0]]
            meter = self.templates[temp].split("_")[-1]
            for e in ends:
                if e not in self.end_pos: self.end_pos[e] = []
                if meter not in self.end_pos[e]: self.end_pos[e].append(meter)


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
        self.gender = random.choice([["i", "me", "my", "mine", "myself"], ["you", "your", "yours", "yourself"],  ["he", "him", "his", "himself"], ["she", "her", "hers", "herself"], ["we", "us", "our", "ours", "ourselves"], ["they", "them", "their", "theirs", "themselves"]])
        #Get rhyming words
        #at some point implement narrative trajectory stuff
        rhyme_dict = {}
        #tone = ['good','good', 'good', 'good', 'bad', 'bad', 'excellent'] #for example
        #for i,j in zip(['A', 'B', 'C', 'D', 'E', 'F', 'G'], tone):
        #    rhyme_dict[i] = self.getRhymes([prompt,j]) #one day pass [prompt, narr]
        for i in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            rhyme_dict[i] = self.get_rhymes([prompt], words=self.words_to_pos.keys())
        last_word_dict = self.last_word_dict(rhyme_dict)
        #for now we shall generate random words, but they will fit the meter, rhyme and templates

        candidates = ["         ----" + prompt.upper() + "----"]
        used_templates = []
        for line_number in range(1,15):
            first_word = random.choice(list(last_word_dict[line_number]))  # last word is decided in last_word_dict
            while not self.get_meter(first_word) or not self.suitable_last_word(first_word): #make sure its valid
                first_word = random.choice(list(last_word_dict[line_number]))
            in_template = self.get_word_pos(first_word)[0]
            while in_template not in self.end_pos or not any(pos in self.end_pos[in_template] for pos in self.dict_meters[first_word]):
                in_template = random.choice(self.get_word_pos(first_word)) #some words have multiple POS so make sure it picks the one with an existing template
            in_meter = [poss_meter for poss_meter in self.dict_meters[first_word] if poss_meter in self.end_pos[in_template]]
            if len(in_meter) < 1:
                print(first_word, in_meter)
                print(1/0) #shouldnt get here, will crash if it does
            in_meter = in_meter[0]
            curr_line = line.Line(first_word + " ", in_meter, pos_template=in_template) #adds space in case punctuation is replaced
            template = False
            while curr_line.syllables < 10: #iterates until line is complete
                #if reset: print("HI", curr_line.text)
                while not template:
                    print("tick:", curr_line.text, curr_line.pos_template, [c.text for c in candidates[1:]])
                    template = self.get_random_template(curr_line.pos_template, curr_line.meter)
                    if not template: input("help" + curr_line.pos_template + curr_line.meter + curr_line.text)
                    #print("a", template)
                    if template[-1] in ",.?;" and curr_line.text[-1] != template[-1]:
                        curr_line.text = curr_line.text[:-1] + template[-1]
                        print("text modified with", template[-1])
                    elif template[-1] in ">" and curr_line.text[-1] not in ",.?;":
                        curr_line.text = curr_line.text[:-1] + random.choice(template.split("<")[-1].strip(">").split("/"))
                        print("text modified with", curr_line.text[-1])


                    if line_number == 14 and template[-1] in ",;>" + string.ascii_lowercase: template = self.get_random_template(curr_line.pos_template, curr_line.meter, end_punc=".")
                    elif len(candidates) > 1 and candidates[-1].text[-1] in ",;" + string.ascii_lowercase:
                        template = self.get_random_template(curr_line.pos_template, curr_line.meter, end_punc=".")
                        print("tempalte modified with .")
                        if not template:
                            template = self.get_random_template(curr_line.pos_template, curr_line.meter, end_punc="?")
                            print("tempalte modified with ?")

                            if not template:
                                print("trying to end ", curr_line.text[:-1], " ", curr_line.pos_template, " ", curr_line.meter, " with a . but not happening")
                                template = self.get_random_template(curr_line.pos_template, curr_line.meter)
                                curr_line.text = curr_line.text[:-1] + "."

                curr_line.text = curr_line.text.strip()

                if template and template.split()[0] in ['AND', 'THAT'] and ((line_number-1)%2 == 0  or len(candidates) > 1 and candidates[-1].text[-1] == "."):
                    print("oi oi", template, line_number, curr_line.text)
                    template = self.get_random_template(curr_line.pos_template, curr_line.meter, exclude=["AND", "THAT"]) #makes sure first line of each stanza doesnt start with AND
                    if not template:
                        curr_line.reset()
                        template = False
                        continue
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
                    #NOT GREAT - shouldnt get here
                    curr_line.syllables = 100
                    curr_line.print_info()
                    print(1/0)
                    continue
                #if template in used_templates: #reduces but doesnt eliminate chance of reusing templates (sometimes have to)
                #    template = self.get_random_template(curr_line.pos_template, curr_line.meter)

                next_pos = template.split()[-len(curr_line.pos_template.split()) - 1] #gets next POS from the right
                next_meter = self.templates[template].split("_")[-len(curr_line.pos_template.split()) - 1] #gets next meter
                poss_words = self.get_pos_words(next_pos, meter=next_meter) #gets all possible words which fit pos and meter
                if not poss_words:
                    print("no words", next_pos, next_meter, template)
                    print(1/0) #if there arent, die

                next_word = random.choice(poss_words) #pick word randomly

                curr_line.add_word(next_word, next_meter) #updates line
                curr_line.pos_template = next_pos + " " + curr_line.pos_template
                #template = False #make a parameter?

            #line finished generating
            print("adding line", line_number)
            curr_line.print_info()
            candidates.append(curr_line)
            used_templates.append(curr_line.pos_template)

        #poem finished generating
        if print_poem:
            print("")
            print(candidates[0])
            del candidates[0]
            for cand in range(len(candidates)):
                print(candidates[cand].text)#, ": ", candidates[cand].meter)
                if( (cand + 1) % 4 == 0): print("")
        #return candidates



    def get_random_template(self, curr_template, curr_meter, pref_pos=None, exclude=None, end_punc=None):
        """
        Gets a random template given the current POS and meter templates
        Parameters
        ----------
        curr_template - current template (from the end), eg NN VBZ
        curr_meter - corresponding meter, eg 10_1
        pref_pos (optional) - a dictionary of POS's and how they should be weighted, e.g {"JJ": 1, "VBD":-1} would be more likely to a return a template with more adjectives and fewer past tense verbs
        exclude (optional)  - a list of POS which you dont want to begin a template with, eg ['AND'] for the first line of a stanza

        Returns - a randomly chosen valid template

        """
        #gets all templates which end in curr_template and curr_meter
        poss_templates = [item for item in self.templates.keys() if item.translate(str.maketrans('', '', string.punctuation))[-len(curr_template):] == curr_template and self.templates[item].split('_')[-len(curr_meter.split('_')):] == curr_meter.split('_')]
        if exclude: poss_templates = [x for x in poss_templates if x.split()[0] not in exclude] #if exclude is given, remove those ones
        if end_punc: poss_templates = [x for x in poss_templates if x[-1] == end_punc or ("<" in x and end_punc in x.split("<")[1].strip(">").split("/"))]
        if len(poss_templates) == 0: return []
        if pref_pos:
            n = len(poss_templates)
            template_scores = np.zeros(n)
            for i in range(n): #iterates through all valid templates
                score = 0
                for pos in poss_templates[i].split(): #iterates through each POS in the template
                    if pos in pref_pos: score += pref_pos[pos] #adds the weight of that POS
                template_scores[i] = score

            return np.random.choice(poss_templates, p=helper.softmax(template_scores)) #Very nifty function which chooses from a list with a custom distribution
        return random.choice(poss_templates)

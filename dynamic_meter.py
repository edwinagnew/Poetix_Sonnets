import numpy as np

import pickle
from collections import defaultdict
from py_files import helper
import random
from py_files import line

import string


import poem_core

class Dynamic_Meter(poem_core.Poem):
    def __init__(self,words_file="saved_objects/tagged_words.p",
                 syllables_file='saved_objects/cmudict-0.7b.txt',
                 wv_file='saved_objects/word2vec/model.txt',
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

        self.meter_and_pos = {}
        self.possible_meters = ["1", "0", "10", "01", "101", "010", "1010", "0101", "10101", "01010", "101010", "010101"]#the possible meters a word could have
        possible_pos = list(self.pos_to_words.keys())
        for pos in possible_pos:
            for meter in self.possible_meters:
                self.meter_and_pos[(meter, pos)] = [word for word in self.pos_to_words[pos] if word in self.dict_meters and meter in self.dict_meters[word]]
        for word in self.special_words:
            if word in self.dict_meters:
                meter = self.dict_meters[word]
                self.meter_and_pos[(meter, word)] = [word]
            else:
                continue

        if prompt:
            self.gen_poem_edwin(prompt)

    def get_poss_meters(self, template, meter): #template is a list of needed POS, meter is a string of the form "0101010..." or whatever meter remains to be assinged
        word_pos = template[-1]
        if len(template) == 1:
            if (meter, word_pos) not in self.meter_and_pos or len(self.meter_and_pos[(meter, word_pos)]) == 0:
                return None
            else:
                return {meter: {}} #should return a list of meters for the next word to take. in base case there is no next word, so dict is empty
        else:
            poss_meters = {}
            for poss_meter in self.possible_meters:
                #print("checking for ", word_pos, "with meter ", poss_meter)
                if poss_meter in meter[:len(poss_meter)] and (poss_meter, word_pos) in self.meter_and_pos and len(self.meter_and_pos[(poss_meter, word_pos)]) > 0:
                    temp = self.get_poss_meters(template[:-1], meter[len(poss_meter):])
                    #print("made recursive call")
                    if temp != None:
                        #print("adding something to dict")
                        poss_meters[poss_meter] = temp
            if len(poss_meters) == 0:
                return None
            return poss_meters


    def check_template(self, template, meter): #makes sure any special words are in the meter dictionary
        for i in range(len(template)):
            if template[i] in self.special_words:
                if (meter[i], template[i]) not in self.dict_meters:
                    print("adding to dictionary the word, ", template[i], "with meter ", meter[i])
                    self.meter_and_pos[(meter[i], template[i])] = [template[i]]
                else:
                    self.meter_and_pos[(meter[i], template[i])].append([template[i]])

    def create_meter_test(self):
        #template = random.choice(list(self.templates.keys()))
        #meter = self.templates[template]
        template = ['VBD', 'AN', 'ALL', 'JJ', 'NN', 'AND', 'JJ', 'NN.']
        meter = self.templates[" ".join(template)]
        if type(template) == str: template = template.split()
        if type(meter) == str: meter = meter.split("_")
        self.check_template(template, meter)
        template = ['VBD', 'AN', 'ALL', 'JJ', 'NN', 'AND', 'JJ', 'NN']
        starting_meter = "1010101010" #we work through the meter backwords, because it makes indexing easier
        print(template)
        poss_meters = self.get_poss_meters(template, starting_meter)
        print(poss_meters)
        new_meters = []
        while len(new_meters) < len(template):
            new_meter = random.choice(list(poss_meters.keys()))
            new_meters.append(new_meter)
            poss_meters = poss_meters[new_meter]
        new_meters.reverse()
        return (template, new_meters)

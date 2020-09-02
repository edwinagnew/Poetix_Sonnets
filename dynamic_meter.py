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

        """
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
        """
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
        for item in self.meter_and_pos["0"]:
            self.meter_and_pos["1"].append(item)
        for item in self.meter_and_pos["1"]:
            self.meter_and_pos["0"].append(item)

    def get_poss_meters(self, template, meter): #template is a list of needed POS, meter is a string of the form "0101010..." or whatever meter remains to be assinged (but backward)
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
        elif word_pos[-1] in [",", ".", ":", ";",">", "?"]:
            word_pos = word_pos[:-1]

        if word_pos == "POS":
            temp = self.get_poss_meters(template[:-1], meter)
            if temp != None:
                return {"": temp}

        if len(template) == 1:
            check_meter = "".join(reversed(meter))
            if (check_meter, word_pos) not in self.meter_and_pos or len(self.meter_and_pos[(check_meter, word_pos)]) == 0:
                return None
            else:
                return {check_meter: {}} #should return a list of meters for the next word to take. in base case there is no next word, so dict is empty
        else:
            poss_meters = {}
            for poss_meter in self.possible_meters:
                #print("checking for ", word_pos, "with meter ", poss_meter)
                check_meter = "".join(reversed(poss_meter)) #we're going through the string backwords, so we reverse it
                if meter.find(poss_meter) == 0 and (check_meter, word_pos) in self.meter_and_pos and len(self.meter_and_pos[(check_meter, word_pos)]) > 0:
                    temp = self.get_poss_meters(template[:-1], meter[len(poss_meter):])
                    #print("made recursive call")
                    if temp != None:
                        #print("adding something to dict")
                        poss_meters[check_meter] = temp
            if len(poss_meters) == 0:
                return None
            return poss_meters


    def get_poss_meters_forward_rhet(self, template, meter, rhet_dict): #template is a list of needed POS, meter is a string of the form "0101010..." or whatever meter remains to be assinged
        """
        :param template: A list of POS's and/or special words
        :param meter: The desired meter for the line, given forward as a string of the form "0101010101".
        rhet_dict: a dictionary where the keys are words that have a required meter, and the values are that meter.
        :return: A dictionary with meters as keys mapping possible meter values for the last word in template to dicts in which the keys are the possible values
        the next word can take on, given that meter assigned to the last word.
        """
        if type(template) == str: template = template.split()

        pair_info = None

        word_pos = template[0]
        if word_pos[-1] == ">":
            word_pos = word_pos.split("<")[0]
        elif word_pos[-1] in [",", ".", ":", ";",">", "?"]:
            word_pos = word_pos[:-1]
        if "_" in word_pos:
            rhet_info = word_pos.split("_")
            word_pos = rhet_info[0]
            pair_info = rhet_info[1]


        if word_pos == "POS":
            temp = self.get_poss_meters_forward_rhet(template[1:], meter, rhet_dict)
            if temp != None:
                return {"": self.get_poss_meters_forward_rhet(template[1:], meter, rhet_dict)}

        if len(template) == 1:
            if (pair_info != None and meter != rhet_dict[pair_info]) or (meter, word_pos) not in self.meter_and_pos or len(self.meter_and_pos[(meter, word_pos)]) == 0:
                return None
            else:
                return {meter: {}} #should return a list of meters for the next word to take. in base case there is no next word, so dict is empty
        else:
            poss_meters = {}
            for poss_meter in self.possible_meters:
                #print("checking for ", word_pos, "with meter ", poss_meter)
                if meter.find(poss_meter) == 0 and (poss_meter, word_pos) in self.meter_and_pos and len(self.meter_and_pos[(poss_meter, word_pos)]) > 0:
                    if pair_info != None and pair_info in rhet_dict and rhet_dict[pair_info] != poss_meter:
                        continue
                    if pair_info != None:
                        temp_dict = rhet_dict.copy()
                        temp_dict[pair_info] = poss_meter
                        temp = self.get_poss_meters_forward_rhet(template[1:], meter[len(poss_meter):], temp_dict)
                    else:
                        temp = self.get_poss_meters_forward_rhet(template[1:], meter[len(poss_meter):], rhet_dict)
                    #print("made recursive call")
                    if temp != None:
                        #print("adding something to dict")
                        poss_meters[poss_meter] = temp
            if len(poss_meters) == 0:
                return None
            return poss_meters

    def get_poss_meters_forward(self, template, meter): #template is a list of needed POS, meter is a string of the form "0101010..." or whatever meter remains to be assinged
        """
        :param template: A list of POS's and/or special words
        :param meter: The desired meter for the line, given forward as a string of the form "0101010101".
        :return: A dictionary with meters as keys mapping possible meter values for the last word in template to dicts in which the keys are the possible values
        the next word can take on, given that meter assigned to the last word.
        """
        if type(template) == str: template = template.split()

        word_pos = template[0]
        if word_pos[-1] == ">":
            word_pos = word_pos.split("<")[0]
        elif word_pos[-1] in [",", ".", ":", ";",">", "?"]:
            word_pos = word_pos[:-1]

        if word_pos == "POS":
            temp = self.get_poss_meters_forward(template[1:], meter)
            if temp != None:
                return {"": self.get_poss_meters_forward(template[1:], meter)}

        if len(template) == 1:
            if (meter, word_pos) not in self.meter_and_pos or len(self.meter_and_pos[(meter, word_pos)]) == 0:
                return None
            else:
                return {meter: {}} #should return a list of meters for the next word to take. in base case there is no next word, so dict is empty
        else:
            poss_meters = {}
            for poss_meter in self.possible_meters:
                #print("checking for ", word_pos, "with meter ", poss_meter)
                if meter.find(poss_meter) == 0 and (poss_meter, word_pos) in self.meter_and_pos and len(self.meter_and_pos[(poss_meter, word_pos)]) > 0:
                    temp = self.get_poss_meters_forward(template[1:], meter[len(poss_meter):])
                    #print("made recursive call")
                    if temp != None:
                        #print("adding something to dict")
                        poss_meters[poss_meter] = temp
            if len(poss_meters) == 0:
                return None
            return poss_meters



    def check_template(self, template, meter): #makes sure any special words are in the meter dictionary, takes (template and meter as lists)
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
                if (meter[i], word) not in self.dict_meters:
                    print("adding to dictionary the word, ", word, "with meter ", meter[i])
                    self.meter_and_pos[(meter[i], word)] = [word]
                else:
                    self.meter_and_pos[(meter[i], word)].append([word])

    def create_meter_test(self):
        """
        This fuction is desined merely to test the above functions, and show how they can be used.
        :return:
        """
        #template = random.choice(list(self.templates.keys()))
        #meter = self.templates[template]
        template = random.choice(list(self.templates.keys()))
        #template = "TO VB THE NN POS NN BY THE NN AND VB"
        meter = self.templates[template]
        print(meter)
        self.check_template(template, meter) #need to make sure any special words are in our meter dict.
        starting_meter = "1010101010" #we work through the meter backwords, because it makes indexing easier
        print(template)
        poss_meters = self.get_poss_meters(template, starting_meter)
        print(poss_meters)
        new_meters = []
        for i in range(len(template.split()) - 1, -1, -1):
            new_meter = random.choice(list(poss_meters.keys()))
            new_meters.append(new_meter)
            poss_meters = poss_meters[new_meter]
        new_meters.reverse()
        print(new_meters)
        print(self.write_line_random(template, new_meters))
        return (template, new_meters)

    def create_meter_test_hardcore(self): #same as above, but for all the templates, to make sure we don't run into errors
        for template in list(self.templates.keys()):
            meter = self.templates[template]
            print(meter)
            self.check_template(template, meter)  # need to make sure any special words are in our meter dict.
            starting_meter = "1010101010"  # we work through the meter backwords, because it makes indexing easier
            print(template)
            poss_meters = self.get_poss_meters(template, starting_meter)
            print(poss_meters)
            new_meters = []

            for i in range(len(template.split()) - 1, -1, -1):
                new_meter = random.choice(list(poss_meters.keys()))
                new_meters.append(new_meter)
                poss_meters = poss_meters[new_meter]
            new_meters.reverse()
            """
            for i in range(len(template.split())):
                new_meter = random.choice(list(poss_meters.keys()))
                new_meters.append(new_meter)
                poss_meters = poss_meters[new_meter]
            """
            print(new_meters)
            print(self.write_line_random(template, new_meters))
            #return (template, new_meters)
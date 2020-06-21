import random
from nltk.corpus import wordnet as wn
from nltk import PorterStemmer
import POS_converter

class Character():
    def __init__(self, type, gender, special_adj, score_function = lambda x: 1):
        """
        Create a character, with pronouns and adjectives, that can thus be set to a role in a story and described consistently.
        It takes a list of reference words in case we need to have access to some other kind of info about the words we choose.
        """

        self.pronouns = {}
        if gender == "male":
            pronouns_to_add = ["him", "his", "he", "himself"]
        if gender == "female":
            pronouns_to_add = ["her", "hers", "she", "herself"]
        if gender == "you":
            pronouns_to_add = ["you", "yours", "thee", "thy", "thine", "thou", "yourself", "thyself"]
        if gender == "none":
            pronouns_to_add = ["it", "its", "itself"]
        for item in pronouns_to_add:
            self.pronouns[item] = 1

        self.char_words = {}
        noun_to_add = []
        meronyms = []
        if type == "forest":
            meronyms = ["trees", "canopy", "undergrowth", "brambles", "roots"]
            noun_to_add += meronyms
        for item in wn.synsets(type):
            if ".n.01" in item.name():
                noun_to_add += wn.synset(item.name()).lemma_names()
                #meronyms = [thing.name().split(".")[0] for thing in wn.synset(item.name()).part_meronyms()]

        for item in noun_to_add:
            self.char_words[item] = score_function(item)
        #print(meronyms)

        self.char_adj = {}
        adj_to_add = []
        for item in wn.synsets(special_adj):
            if ".s." in item.name():
                adj_to_add += wn.synset(item.name()).lemma_names()
        for item in adj_to_add:
            self.char_adj[item] = score_function(item)

        pos_conv = POS_converter.POS_changer()

        #print(pos_conv.close_adv('happy'))
        
        self.char_adv = {}
        adv_to_add = []
        base_adv = pos_conv.close_adv(special_adj)[0][0]

        for item in wn.synsets(base_adv):
            if ".r." in item.name():
                adv_to_add += wn.synset(item.name()).lemma_names()
        for item in adv_to_add:
            self.char_adv[item] = score_function(item)


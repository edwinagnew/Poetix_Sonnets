import random
from nltk.corpus import wordnet as wn
from nltk import PorterStemmer

class Character():
    def __init__(self, type, gender, special_adj, score_function = lambda x: 1):
        """
        Create a character, with pronouns and adjectives, that can thus be set to a role in a story and described consistently.
        It takes a list of reference words in case we need to have access to some other kind of info about the words we choose.
        """

        self.pronouns = {}
        if gender == "male":
            pronouns_to_add = ["him", "his", "he"]
        if gender == "female":
            pronouns_to_add = ["her", "hers", "she"]
        if gender == "you":
            pronouns_to_add = ["you", "yours", "thee", "thy", "thine", "thou"]
        for item in pronouns_to_add:
            self.pronouns[item] = 1

        self.char_words = {}
        noun_to_add = []
        for item in wn.synsets(type):
            if ".n.01" in item.name():
                noun_to_add += wn.synset(item.name()).lemma_names()
        for item in noun_to_add:
            self.char_words[item] = score_function(item)

        self.char_adj = {}
        adj_to_add = []
        for item in wn.synsets(special_adj):
            if ".s." in item.name():
                adj_to_add += wn.synset(item.name()).lemma_names()
        for item in adj_to_add:
            self.char_adj[item] = score_function(item)
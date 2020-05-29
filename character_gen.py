import random
from nltk.corpus import wordnet as wn
from nltk import PorterStemmer

class Character():
    def __init__(self, type, gender, special_adj):
        """
        Create a character, with pronouns and adjectives, that can thus be set to a role in a story and described consistently.
        It takes a list of reference words in case we need to have access to some other kind of info about the words we choose.
        """


        if gender == "male":
            self.pronouns = ["him", "his", "he"]
        if gender == "female":
            self.pronouns = ["her", "hers", "she"]
        if gender == "you":
            self.pronouns = ["you", "yours", "thee", "thy", "thine", "thou"]
        #syn = wn.synsets(type)

        self.char_words = []
        for item in wn.synsets(type):
            if ".n.01" in item.name():
                self.char_words += wn.synset(item.name()).lemma_names()

        self.char_adj = []
        for item in wn.synsets(special_adj):
            if ".s." in item.name():
                self.char_adj += wn.synset(item.name()).lemma_names()

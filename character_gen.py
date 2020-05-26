import random
from nltk.corpus import wordnet as wn
from nltk import PorterStemmer

class Character():
    def __init__(self, type, gender, alignment, reference = None):
        """
        Create a character, with pronouns and adjectives, that can thus be set to a role in a story and described consistently.
        It takes a list of reference words in case we need to have access to some other kind of info about the words we choose.
        """


        if gender == "male":
            self.pronouns = ["him", "his", "he"]
        if gender == "female":
            self.pronouns = ["her", "hers", "she"]
        if alignment == "good":
            self.char_adj = ["valiant", "brave", "strong", "kind", "generous", "noble"]
        if alignment == "bad":
            self.char_adj = ["wicked", "cruel", "mean", "greedy", "avaricious", "vile"]
        syn = wn.synsets(type)
        self.char_words = syn[0].lemma_names()
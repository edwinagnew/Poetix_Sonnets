#import io
#from gensim.models.fasttext import FastText as FT_gensim
#from gensim.test.utils import datapath
from gensim.models.keyedvectors import KeyedVectors
import pickle


class Sim_finder:
    def __init__(self, model_file="saved_objects/fasttext/wiki-news-300d-1M.vec", pick_file="saved_objects/fasttext/model.p"):
        try:
            self.model = pickle.load(open(pick_file, "rb"))
            print("loaded fasttext from pickle")
        except:
            print("loading fasttext for the first time")
            self.model = KeyedVectors.load_word2vec_format(model_file)
            print("saving with pickle")
            pickle.dump(self.model, open(pick_file, "wb"))


    def gensim_test():
        model = KeyedVectors.load_word2vec_format("wiki-news-300d-1M.vec")
        print(model.most_similar("nights"))

    def get_close_words(self, positive, negative=None, n=50):
        if type(positive) == str: positive = positive.split()
        return [item[0] for item in self.model.most_similar(positive, negative, topn=n)]

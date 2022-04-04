#import io
#from gensim.models.fasttext import FastText as FT_gensim
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from difflib import SequenceMatcher
import pickle


class Sim_finder:
    def __init__(self, ft_model_file="saved_objects/fasttext/wiki-news-300d-1M.vec", ft_pick_file="saved_objects/fasttext/model.p", glove_model_file="saved_objects/glove/glove.42B.300d.txt", glove_pick_file="saved_objects/glove/model.p"):
        try:
            self.fasttext_model = pickle.load(open(ft_pick_file, "rb"))
            print("loaded fasttext from pickle")
        except:
            print("loading fasttext for the first time")
            self.fasttext_model = KeyedVectors.load_word2vec_format(ft_model_file)
            print("saving with pickle")
            pickle.dump(self.fasttext_model, open(ft_pick_file, "wb"))

        try:
            self.glove_model = pickle.load(open(glove_pick_file, "rb"))
            print("loaded glove from pickle")
        except:
            print("loading glove for the first time")
            glove_file = glove_model_file
            tmp_file = get_tmpfile("glove_word2vec.txt")
            glove2word2vec(glove_file, tmp_file)
            self.glove_model = KeyedVectors.load_word2vec_format(tmp_file, binary=False)
            print("saving with pickle")
            pickle.dump(self.glove_model, open(glove_pick_file, "wb"))


    def gensim_test():
        model = KeyedVectors.load_word2vec_format("wiki-news-300d-1M.vec")
        print(model.most_similar("nights"))

    def get_close_words(self, positive, negative=None, n=50):
        if type(positive) == str: positive = positive.split()
        return [item[0] for item in self.fasttext_model.most_similar(positive, negative, topn=n)]

    def ft_word_similarity(self, word1, word2, choice=max):
        if type(word1) == list: return choice(self.ft_word_similarity(w, word2) for w in word1)
        if type(word2) == list: return choice(self.ft_word_similarity(word1, w) for w in word2)

        if word1 not in self.fasttext_model.vocab or word2 not in self.fasttext_model.vocab: return 0
        return self.fasttext_model.similarity(word1, word2)

    def gl_word_similarity(self, word1, word2, choice=max):
        if type(word1) == list: return choice(self.gl_word_similarity(w, word2) for w in word1)
        if type(word2) == list: return choice(self.gl_word_similarity(word1, w) for w in word2)

        if word1 not in self.glove_model.vocab or word2 not in self.glove_model.vocab: return 0
        return self.glove_model.similarity(word1, word2)

    def both_similarity(self, word1, word2, choice=min):
        return choice(self.ft_word_similarity(word1, word2), self.gl_word_similarity(word1, word2))

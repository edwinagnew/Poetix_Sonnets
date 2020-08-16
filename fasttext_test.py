import io
from gensim.models.fasttext import FastText as FT_gensim
from gensim.test.utils import datapath
from gensim.models.keyedvectors import KeyedVectors


class Sim_finder:
    def __init__(self, model_file="wiki-news-300d-1M.vec"):
        self.model = KeyedVectors.load_word2vec_format(model_file)

    def load_vectors(fname):
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = map(float, tokens[1:])
        return data

    def gensim_test():
        model = KeyedVectors.load_word2vec_format("wiki-news-300d-1M.vec")
        print(model.most_similar("nights"))

    def get_close_words(self, positive, negative=None):
        return [item[0] for item in self.model.most_similar(positive, negative, topn=50)]

"""
if __name__ == "__main__":
    #model = load_vectors("wiki-news-300d-1M.vec")
    #print(model)
    gensim_test()
"""
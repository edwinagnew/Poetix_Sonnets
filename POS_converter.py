from gensim.models.keyedvectors import KeyedVectors
from difflib import SequenceMatcher
from nltk.corpus import wordnet as wn
import pickle
import poem_core


class POS_changer():

    def __init__(self, model_file="saved_objects/fasttext/wiki-news-300d-1M.vec", pick_file="saved_objects/fasttext/model.p"):
        try:
            self.model = pickle.load(open(pick_file, "rb"))
            print("loaded fasttext from pickle")
        except:
            print("loading fasttext for the first time")
            self.model = KeyedVectors.load_word2vec_format(model_file)
            print("saving with pickle")
            pickle.dump(self.model, open(pick_file, "wb"))
        self.poem = poem_core.Poem()


    def close_adv(self, input, num=5, model_topn=50):
        if type(input) == str:
            positive = input.split() + ['happily']
        else:
            positive = input + ["happily"]
        negative = [       'happy']
        all_similar = self.model.most_similar(positive, negative, topn=model_topn)

        def score(candidate):
            ratio = SequenceMatcher(None, candidate, input).ratio()
            looks_like_adv = 1.0 if candidate.endswith('ly') else 0.0
            return ratio + looks_like_adv

        close = sorted([(word, score(word)) for word, _ in all_similar], key=lambda x: -x[1])
        return close[:num]

    def close_jj(self, input, num=5, model_topn=50):
        #positive = [input, 'dark']
        negative = [       'darkness']
        if type(input) == str:
            positive = input.split() + ['dark']
        else:
            positive = input + ["dark"]
        all_similar = self.model.most_similar(positive, negative, topn=model_topn)
        close = [word[0] for word in all_similar if word[0] in self.poem.pos_to_words["JJ"]]

        return close

    def close_jj_from_verb(self, input, num=5, model_topn=50):
        positive = [input, 'burnt']
        negative = [       'burn']
        all_similar = self.model.most_similar(positive, negative, topn=model_topn)
        close = [word[0] for word in all_similar if word[0] in self.poem.pos_to_words["JJ"]]

        return close

    def nltk_JJ(self, wordtoinv):
        s = []
        winner = ""
        for ss in wn.synsets(wordtoinv):
            for lemmas in ss.lemmas():  # all possible lemmas.
                s.append(lemmas)

        for pers in s:
            posword = pers.pertainyms()[0].name()
            if posword[0:3] == wordtoinv[0:3]:
                winner = posword
                break

        return winner

    def close_nn(self, input, num=5, model_topn=50):
        negative = ['dark']
        if type(input) == str:
            positive = input.split() + ['darkness']
        else:
            positive = input + ["darkness"]
        all_similar = self.model.most_similar(positive, negative, topn=model_topn)
        close = [word[0] for word in all_similar if word[0] in self.poem.pos_to_words["NN"] or word[0] in self.poem.pos_to_words["NNS"]]

        return close

    def close_vb(self, input, num=5, model_topn=50):
        positive = input
        all_similar = self.model.most_similar(positive, topn=model_topn)
        close = [word[0] for word in all_similar if
                 word[0] in self.poem.pos_to_words["VB"] or word[0] in self.poem.pos_to_words["VBP"] or
                 word[0] in self.poem.pos_to_words["VBD"] or word[0] in self.poem.pos_to_words["VBN"]or
                 word[0] in self.poem.pos_to_words["VBZ"] or word[0] in self.poem.pos_to_words["VBG"]]

        return close
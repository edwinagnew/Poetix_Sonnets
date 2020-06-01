from gensim.models.keyedvectors import KeyedVectors
from difflib import SequenceMatcher

class POS_changer():

    def __init__(self, glove_filename = 'glove-word2vec.6B.100d.txt'):
        self.model = KeyedVectors.load_word2vec_format(glove_filename, binary=False)

    def close_adv(self, input, num=5, model_topn=50):
        positive = [input, 'happily']
        negative = [       'happy']
        all_similar = self.model.most_similar(positive, negative, topn=model_topn)

        def score(candidate):
            ratio = SequenceMatcher(None, candidate, input).ratio()
            looks_like_adv = 1.0 if candidate.endswith('ly') else 0.0
            return ratio + looks_like_adv

        close = sorted([(word, score(word)) for word, _ in all_similar], key=lambda x: -x[1])
        return close[:num]


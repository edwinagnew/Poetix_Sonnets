from datasets import load_dataset

from spacy.lang.en.stop_words import STOP_WORDS
import string
import gpt_revised
import pickle
import poem_core


def remove_punc(s):
    punc = ".,;:!?-'"
    for p in punc:
        if s.find(p) != -1:
            s = s[:s.find(p)]
    return s


def normalize(count_dict):

    normalized = {}
    for key in count_dict:
        dist = count_dict[key]
        prob = (dist[1] + .5 * dist[3]) / sum(dist)
        normalized[key] = prob
    return normalized

def tokenize_words(gpt, array):
    return [gpt.tokenizer.encode(word) for word in array]


if __name__ == "__main__":
    p_c = poem_core.Poem()
    gpt = gpt_revised.gpt_gen(sonnet_object=p_c)


    dataset = load_dataset("poem_sentiment")


    count_dict = {}

    print("iterating")

    for i, row in enumerate(dataset['train']):

        #remove stopwords
        text = [remove_punc(word) for word in row['verse_text'].split() if remove_punc(word) not in STOP_WORDS]
        encoded = tokenize_words(gpt, text)
        for tokens in encoded:
            for token in tokens:
                if token not in count_dict:
                    count_dict[token] = [0,0,0,0]
                if token:
                    count_dict[token][row['label']] += 1

    final = normalize(count_dict)
    print(final)

    pickle.dump(final, open("saved_objects/bayes_token.p", "wb"))

from datasets import load_dataset

from spacy.lang.en.stop_words import STOP_WORDS
import string


import poem_core


def remove_punc(s):
    punc = ".,;:!?-"
    for p in punc:
        if s.find(p) != -1:
            s = s[:s.find(p)]
    return s

p_c = poem_core.Poem()




dataset = load_dataset("poem_sentiment")


count_dict = {}

print("iterating")

for i, row in enumerate(dataset['train']):

    #remove stopwords
    text = [remove_punc(word) for word in row['verse_text'].split() if word not in STOP_WORDS]


    for word in text:
        if word not in count_dict:
            count_dict[word] = [0,0,0,0]


        count_dict[word][row['label']] += 1


print(count_dict)
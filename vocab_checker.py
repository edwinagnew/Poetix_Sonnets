import os
import string
import pickle

import numpy as np

import nltk
import progressbar

from lxml.html import parse

rem = string.punctuation + "0123456789" + "“‘-”—’"
translator = str.maketrans(rem, ' ' * len(rem))

word_set = set()
for folder in os.listdir(os.getcwd() + "/word_corpus"):
    if folder == "wikipedia": continue
    for file in os.listdir(os.getcwd() + "/word_corpus/" + folder):
        f = open(os.getcwd() + "/word_corpus/" + folder + "/" + file)
        print("reading ", f.name)
        text = f.read()
        word_set.update(text.translate(translator).lower().strip("'").split())

#print(word_set)
print(len(word_set))

postag_file='saved_objects/postag_dict_all+VBN.p'
with open(postag_file, 'rb') as f:
    postag_dict = pickle.load(f)
pos_to_words = postag_dict[1]
words_to_pos = postag_dict[2]

#word_set.update(words_to_pos.keys())
word_set.update(" ".join(words_to_pos.keys()).translate(translator).lower().split())

print(len(word_set), "(o" in word_set)
#tag wikipedia + dickens and store in histograms

print("reading wiki file...")

#tree = parse("wiki_file.xml")
#tree = parse("enwiki-20181001-corpus.xml")
#root = tree.getroot()
#wiki_text = root.text_content().lower()
wi = open("text_wiki.txt", "r")
wiki_text = wi.read().lower()


print("reading imdb files")
imdb_text = ""
for folder in os.listdir(os.getcwd() + "/aclImdb/"):
        for sub_folder in os.listdir(os.getcwd() + "/aclImdb/" + folder):
                for file in os.listdir(os.getcwd() + "/aclImdb/" + folder + "/" + sub_folder):
                        f = open(os.getcwd() + "/aclImdb/" + folder + "/" + sub_folder + "/" +  file)
                        print("reading ", f.name)
                        text = f.read()
                        imdb_text += text.lower()
print("reading dickens text")
dick_text = ""
for file in os.listdir(os.getcwd() + "/word_corpus/dickens"):
    f = open(os.getcwd() + "/word_corpus/dickens/" + file)
    dick_text += f.read().lower()

sentences = (wiki_text + " . " + dick_text + " . " + imdb_text).replace("'", " ").split(".")
#sentences = wiki_text.split(".")
print("evaluating ", len(sentences), " sentences")
tagged_words = {}
for sentence in progressbar.progressbar(sentences):
    tags = nltk.pos_tag(sentence.split())
    for tag in tags:
        word = tag[0]
        pos = tag[1]
        if word in word_set and "'" not in word:
            if word not in tagged_words: tagged_words[word] = {}
            if pos not in tagged_words[word]: tagged_words[word][pos] = 0
            tagged_words[word][pos] += 1
print(tagged_words)

for word in list(tagged_words):
    total = sum(tagged_words[word].values(), 0.0)
    if total < 3:
        print(word, "not found enough -- deleted")
        del tagged_words[word]
    elif total <= 10 and len(tagged_words[word]) != 1:
        print(word, tagged_words[word], "not consistent enough --deleted")
        del tagged_words[word]
    else:
        tagged_words[word] = {k: v / total for k, v in tagged_words[word].items()}
        while min(tagged_words[word].values()) < 0.1:
            print("\nbefore: ", word, tagged_words[word])
            val, mindex = min((tagged_words[word][val], val) for (idx, val) in enumerate(tagged_words[word]))
            print("deleted ", tagged_words[word][mindex], "from ", word)
            del tagged_words[word][mindex]
            print("after:", word, tagged_words[word] , "\n")
#input("ready?")
print(tagged_words)

pickle.dump(tagged_words, open("saved_objects/tagged_words.p", "wb"))
print("finished")
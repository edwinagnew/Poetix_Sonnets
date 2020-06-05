"""print("loading...")
import pickle
special_words = {'WHAT', 'MORE', 'EXCEPT', 'WITHOUT', 'ASIDE', 'WHY',
     'AWAY', 'OF', 'COULD', 'WHOSOEVER', 'WHENEVER', 'SHALL', 'ALBEIT',
     'FROM', 'BETWEEN', 'CAN', 'HOW', 'OUT', 'CANNOT',
     'SO', 'BACK', 'ABOUT', 'LATER', 'IF', 'REGARD',
     'MANY', 'TO', 'THERE', 'UNDER', 'APART',
     'QUITE', 'LIKE', 'WHILE', 'AS', 'WHOSE',
     'AROUND', 'NEITHER', 'WHOM', 'SINCE', 'ABOVE', 'THROUGH', 'ALL',
     'AND', 'SOME', 'MAY', 'HALF', 'WHATEVER', 'BEHIND',
     'BEYOND', 'WHERE', 'SUCH', 'YET', 'UNTO', 'BY', 'NEED',
     'A', 'DURING', 'AT', 'AN', 'OUGHT',
     'BUT', 'DESPITE', 'SHOULD', 'THOSE', 'FOR', 'WHEREVER', 'WHOLE', 'THESE',
     'WHOEVER', 'WITH', 'TOWARD', 'WHICH',
     'BECAUSE', 'WHETHER', 'ONWARD', 'UPON', 'JUST', 'ANY',
     'NOR', 'THROUGHOUT', 'OFF', 'EVERY', 'UP', 'NEXT', 'THAT', 'WOULD',
     'WHATSOEVER', 'AFTER', 'ONTO', 'BESIDE', 'ABOARD', 'OVER', 'BENEATH',
     'INSIDE', 'WHEN', 'OR', 'MUST', 'AMONG', 'MIGHT', 'NEAR', 'PLUS', 'UNTIL',
     'ALONG', 'INTO', 'BOTH', 'EITHER', 'WILL', 'IN',
     'EVER', 'ON', 'AGAINST', 'EACH', 'BELOW',
     'DOWN', 'BEFORE', 'THE', 'WHICHEVER', 'WHO', 'PER', 'THIS',
     'ACROSS', 'THAN', 'WITHIN', 'NOT'}

postag_file = 'saved_objects/postag_dict_all+VBN.p'

with open(postag_file, 'rb') as f:
    postag_dict = pickle.load(f)
pos_to_words = postag_dict[1]
words_to_pos = postag_dict[2]

checked_file = open("saved_objects/verified_vocab.txt", "a")
file = open("saved_objects/verified_vocab.txt", "r")
lines = file.readlines()
already_checked = [l.split()[0] for l in lines]
file.close()
deleted_file = open("saved_objects/deleted_words.txt", "a")
deleted_words = open("saved_objects/deleted_words.txt", "r").read().split("\n")
already_checked += deleted_words


print("loaded")
print("if you dont think a word should be included in the vocabulary, mark it as wrong and then enter nothing when it asks for pos")
print("type 'quit' to quit")
k = 10


def get_word_pos(word):

    # Special case
    if word.upper() in special_words:
        return [word.upper()]
    if word not in words_to_pos:
        return []
    return words_to_pos[word]

def check_pos(pos):
    words = pos_to_words[pos]
    j = 0
    #for i in range(0, len(words), k):
    while j < len(words):
        #j = i
        count = 0
        checking = []
        while count < k and j < len(words):
            if words[j] not in already_checked:
                checking.append(j)
                print(j, ". ", words[j], " : ", " ".join(set(get_word_pos(words[j]))))
                count += 1
            j += 1
        wrong = input("are any of those wrong? Type the corresponding numbers, e.g. '2 3 5' . Otherwise, leave blank. : ")
        if wrong == "quit": return "quit"
        else:
            for h in checking:
                p = " ".join(set(get_word_pos(words[h])))
                if str(h) in wrong.split():
                    p = input("What should the pos for '" + words[h] + "' be? If multiple, separate with spaces. : ")
                if p != "" and words[h] not in already_checked:
                    if wrong == quit: input("here somehow" + words[h] + p)
                    checked_file.write(words[h] + " " + p + "\n")

                elif p == "":
                    deleted_file.write(words[h] + "\n")
                already_checked.append(words[h])
        print("you are ", round(j/len(words) * 100), "% of the way through ", pos, " words")
    return "done"


for pos in pos_to_words:
    if check_pos(pos) == "quit": break




print("saving and quitting")
checked_file.close()
deleted_file.close()
"""
import os
import string
import pickle

translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

word_set = set()
for folder in os.listdir(os.getcwd() + "/word_corpus"):
    if folder == "wikipedia": continue
    for file in os.listdir(os.getcwd() + "/word_corpus/" + folder):
        f = open(os.getcwd() + "/word_corpus/" + folder + "/" + file)
        print("reading ", f.name)
        text = f.read()
        word_set.update(text.translate(translator).lower().split())

#print(word_set)
print(len(word_set))

postag_file='saved_objects/postag_dict_all+VBN.p'
with open(postag_file, 'rb') as f:
    postag_dict = pickle.load(f)
pos_to_words = postag_dict[1]
words_to_pos = postag_dict[2]

word_set.update(words_to_pos.keys())
print(len(word_set))
#tag wikipedia + dickens and store in histograms
import numpy as np

import nltk
import progressbar

from lxml.html import parse

tree = parse("word_corpus/wikipedia/enwiki-20181001-corpus.xml")
root = tree.getroot()
wiki_text = root.text_content()

dick_text = ""
for file in os.listdir(os.getcwd() + "/word_corpus/dickens"):
    f = open(os.getcwd() + "/word_corpus/dickens/" + file)
    dick_text += f.read()

sentences = (wiki_text + " . " + dick_text).replace("' ", " ").split(".")
#sentences = wiki_text.split(".")
print("evaluating ", len(sentences), " sentences")
tagged_words = {}
for sentence in progressbar.progressbar(sentences):
    tags = nltk.pos_tag(sentence.split())
    for tag in tags:
        word = tag[0]
        pos = tag[1]
        if word in word_set:
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

pickle.dump(tagged_words, open("saved_objects/tagged_words.p", "w"))
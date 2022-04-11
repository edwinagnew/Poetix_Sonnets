import requests
import spacy
import poem_core

import pickle

nlp = spacy.load('en')

print("loading...")

def get_trans(response):
    jsons = response.json()
    t = set([])
    for j in jsons:
        if 'def' in j:
            for m in j['def']:
                if 'vd' in m:
                    t.add(m['vd'])
    return list(t)


from spacy.lemmatizer import Lemmatizer, ADJ, NOUN, VERB
lemmatizer = nlp.vocab.morphology.lemmatizer

poem = poem_core.Poem()

#t_dict = {}


t_dict = pickle.load(open("saved_objects/verb_trans.p", "rb"))


orig_counts = len(t_dict)

flag = False

#for pos in poem.pos_to_words.keys():
for pos in ['VBD', 'VBZ', 'VBP', 'VBN', 'VBG', 'VB']:
    if flag:
        break
    if "VB" not in pos:
        continue
    print("checking", pos)
    for word in poem.get_pos_words(pos):
        lemmas = lemmatizer(word, VERB)
        if len(lemmas) > 1:
            lemmas = [l for l in lemmas if l in poem.words_to_pos]
            if len(lemmas) != 1:
                print("bad set of lemmas:", word, lemmatizer(word, VERB))
                continue
        lemma = lemmas[0]
        #print(word, lemma)
        if lemma not in t_dict and word not in t_dict:
            try:
                response = requests.get("https://www.dictionaryapi.com/api/v3/references/collegiate/json/"  + lemma + "?key=7fa7f0b1-accd-4175-81f8-a449cea08ff9")
                ret = get_trans(response)
                if len(ret) > 0:
                    orig_counts -= 1
                    t_dict[lemma] = ret
            except:
                pickle.dump(t_dict, open("saved_objects/verb_trans.p", "wb"))
                print(lemma, "broke so saved")
                flag = True
                break


        counts = len(t_dict) - orig_counts
        #if counts >= 960:
            #print("out of counts")
            #break

print("ended with", len(t_dict), "verbs!")
pickle.dump(t_dict, open("saved_objects/verb_trans.p", "wb"))
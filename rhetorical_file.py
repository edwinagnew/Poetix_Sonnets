import poem_core

import random
import numpy as np

from py_files import helper

import time
import string
import pronouncing

class Gen_Device(poem_core.Poem):
    def __init__(self,
                 syllables_file='saved_objects/cmudict-0.7b.txt',
                 extra_stress_file='saved_objects/edwins_extra_stresses.txt',
                 top_file='saved_objects/words/top_words.txt',
                 templates_file='poems/jordan_templates.txt',
                 #templates_file='poems/number_templates.txt'
                 ):


        poem_core.Poem.__init__(self, words_file="saved_objects/tagged_words.p", templates_file=templates_file,
                                syllables_file=syllables_file, extra_stress_file=extra_stress_file, top_file=top_file)

    def write_line(self,template=None, alliteration=0, sibilance=0, assonance=0, in_rhyme=0, k=1):
        """

        Parameters
        ----------
        template
        rhetorical devices:
            alliteration - weights repetition of the first k letters
            sibilance - weights 's' being first letter in word
            assonance - weights repetition of

        Returns
        -------

        """
        if not template:
            pos_template, meter = random.choice(self.templates)
        elif type(template) == int:
            pos_template, meter = self.templates[template]
        else:
            pos_template, meter = template

        pos_template = pos_template.split()
        meter = meter.split("_")

        remove_cons = str.maketrans("qwrtypsdfghjklzxcvbnm", ' ' * len("qwrtypsdfghjklzxcvbnm"))

        print("going with", pos_template, meter)
        line = ""
        for i in range(len(pos_template)):
            a = time.time()
            poss = self.get_pos_words(pos_template[i], meter=meter[i])
            scores = np.ones(len(poss))
            if random.random() < alliteration and len(poss) > 1:
                sub_line = " ".join(x[:k] for x in line.split())
                scores = [scores[p] * word_similarity(poss[p][:k], sub_line) for p in range(len(poss))]
                print("alliteration", poss[np.argmax(scores)], max(scores), "min:", min(scores), "len: ", len(scores), [y for y in zip(poss, scores) if y[1] > 0])

            if random.random() < sibilance and len(poss) > 1:
                scores = [scores[p] * int(poss[p][0] == 's') for p in range(len(poss))]
                print("sibilance", poss[np.argmax(scores)], max(scores), "min:", min(scores), "len: ", len(scores), [y for y in zip(poss, scores) if y[1] > 0])

            if random.random() < assonance and len(poss) > 1:
                sub_line = line.translate(remove_cons)
                scores = [scores[p] * word_similarity(poss[p].translate(remove_cons), sub_line) for p in range(len(poss))]
                print("assonance", poss[np.argmax(scores)], max(scores), "min:", min(scores), "len: ", len(scores), [y for y in zip(poss, scores) if y[1] > 0])

            if random.random() < in_rhyme and len(poss) > 1:
                scores = [scores[p] * rhyme_count(poss[p], line) for p in range(len(poss))]
                print("internal rhyme", poss[np.argmax(scores)], max(scores), "min:", min(scores), "len: ", len(scores), [y for y in zip(poss, scores) if y[1] > 0])

            #c = time.time()
            dist = helper.softmax(scores, exclude_zeros=True)
            #e = time.time()
            #print("dist after", e-c)
            word = np.random.choice(poss, p=dist)
            print("chose ", word, "with prob", dist[poss.index(word)])
            line += word + " "
            b = time.time()
            print(word, i, "took ", b-a, "\n")

        print(line)


def word_similarity(word, line):
    if not line or len(line) == 0: return 1
    score = 0
    for w in set(word.split()):
        score += line.count(w)
    #for letter in line:
    #    if letter in word: score += 1
    return score#/len(line.split())

def rhyme_count(word, line):
    rhymes = pronouncing.rhymes(word)
    score = 0
    for w in line.split():
        if w in rhymes: score += 1
    return score

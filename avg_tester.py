import iterative_improve
import random
import pickle

with open("saved_objects/w2v.p", "rb") as pickle_in:
    poetic_vectors = pickle.load(pickle_in)


i = iterative_improve.Sonnet_Improve()

word = input("What is the first word? type rand for random: ")

scores = {'mean': 0, 'median': 0, 'max': 0}
while word != "-1":

    if word == "rand": word = random.choice(poetic_vectors)

    mets = {}
    avs = ['mean', 'median', 'max']
    random.shuffle(avs)
    for a in avs:
        #mets[a] = i.generate_metaphor(word, avg=a, verbose=False)
        met = i.generate_metaphor(word, avg=a, verbose=False)
        if met not in mets.keys():
            mets[met] = []
        mets[met].append(a)
    print(word , ": ", [key for key in mets.keys()])
    fav = input("best = ").split()

    if fav:
        for best in fav:
            for avg in mets[best]:
                print(avg, " was in there somewhere")
                scores[avg] += 1

    rand = random.choice(poetic_vectors)
    print("r would be",  rand)
    word = input("What is the next word? ")
    if word == "r": word = rand



print(scores)
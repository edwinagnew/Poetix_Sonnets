from datetime import datetime
import scenery
import random
import pickle

sc = scenery.Scenery_Gen()

poems = []
date = datetime.today().date()
file = open("poems_" + str(date) + ".txt", "a")

for i in range(5):
    theme = random.choice(['love', 'death', 'forest', 'wisdom'])
    b = random.choice([5, 7])
    k = random.choice([3, 5, 7])
    #weight_repetition = random.choice([-1])
    weight_pen = random.choice([1, 1.25, 1.5, 2, 3])
    print("\n\ngenerating poem", i, theme, k, b, weight_pen)

    p = sc.write_poem_revised(theme, verbose=False, b=b, k=k, theme_lines="stanza", rep_penalty=weight_pen)

    print(p)
    poems.append(p)
    file.write("\n\n" + p)
    pickle.dump(poems, open("new_gen_poems/poems_" + str(date) + ".p", "wb"))

file.write("\n\n".join(poems))
print("done")

file.close()

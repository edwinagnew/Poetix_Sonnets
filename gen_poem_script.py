from datetime import datetime
import scenery
import random


sc = scenery.Scenery_Gen()

poems = []
date = datetime.today().date()
file = open("poems_" + str(date) + ".txt", "a")

for i in range(2):
    theme = random.choice(['love', 'death', 'forest', 'wisdom'])
    b = random.choice([5, 7])
    k = random.choice([3, 5, 7])
    weight_repetition = random.choice([-1])
    #theme_threshold = random.choice([0.5, 0.75])
    print("\n\ngenerating poem", i, theme, k, b, weight_repetition)

    p = sc.write_poem_revised(theme, verbose=False, b=b, k=k, theme_lines="stanza", weight_repetition=weight_repetition)

    print(p)
    poems.append(p)
    file.write("\n\n" + p)


file.write("\n\n".join(poems))
print("done")

file.close()

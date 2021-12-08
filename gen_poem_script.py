from datetime import datetime
import scenery
import random


sc = scenery.Scenery_Gen()

poems = []
date = datetime.today().date()
file = open("poems_" + str(date) + ".txt", "a")

for i in range(10):
    theme = random.choice(['love', 'death', 'peace', 'war', 'forest', 'darkness', 'wisdom'])
    b = random.choice([5, 7, 10])
    k = random.choice([2, 3, 5, 7])
    weight_repetition = random.choice([-1, -2, -5])
    theme_threshold = random.choice([0.5, 0.75])
    print("\n\ngenerating poem", i, theme, k, b, weight_repetition, theme_threshold)

    p = sc.write_poem_revised(theme, verbose=False, b=b, k=k, theme_lines="stanza", weight_repetition=weight_repetition,
                              theme_threshold=theme_threshold)

    print(p)
    poems.append(p)
    file.write("\n\n" + p)


file.write("\n\n\n\n\n".join(poems))
print("done")
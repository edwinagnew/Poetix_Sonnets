import scenery
import numpy as np

sc = scenery.Scenery_Gen()

lines = []

for b in range(1,6):
    print("generating", b)
    poem = sc.write_poem_revised(theme="love", k=1, b=b, gpt_size="custom fine_tuning/twice_retrained", tense="present", verbose=False)
    lines.append(poem.split("\n")[1:])
    #print(lines)


print(lines)
print("scoring")

for b in range(1, 6):
    scores = [sc.gpt.score_line(lines[b]) for l in lines[b] if len(l.strip()) > 2]
    print("scores: ", b, "mean: ", np.mean(scores), "var", np.var(scores), "min", np.min(scores), "max", np.max(scores))
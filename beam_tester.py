import scenery
import numpy as np

sc = scenery.Scenery_Gen()

lines = []

for b in range(1,10):
    print("generating", b)
    poem = sc.write_poem_revised(theme="love", k=1, branching=b, gpt_size="custom fine_tuning/twice_retrained", tense="present", verbose=False)
    lines.append([poem.split("\n")[1:]])

print("scoring")
for b in range(1, 10):
    scores = [sc.gpt.score_line(l) for l in lines[b] if len(line.strip()) > 2]
    print("scores: ", b, "mean: ", np.mean(scores), "var", np.var(scores), "min", np.min(scores), "max", np.max(scores))
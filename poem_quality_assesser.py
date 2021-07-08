import scenery
import gpt_revised
import pandas as pd

if __name__ == "__main__":
    filename = "poems/poems_with_retrained.txt"
    sc = scenery.Scenery_Gen()
    sc.gpt = gpt_revised.gpt_gen(sonnet_object=sc, model="custom fine_tuning/twice_retrained")
    f = open(filename)
    lines = [line.strip() for line in f.readlines()]
    scores = [float(sc.gpt.score_line(line)) for line in lines]
    score_dict = {"lines": lines, "scores": scores}
    df = pd.DataFrame.from_dict(score_dict)



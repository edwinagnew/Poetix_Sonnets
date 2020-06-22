import sonnet_basic

import bert_verb
import sonnet_basic
import bulk_line_generator

if __name__ == '__main__':
    s = bert_verb.Scenery_Gen(model="gpt_2")
    generator = bulk_line_generator.Bulk_Gen()
    #sonnet = generator.write_bulk_lines(1000, "death")
    sonnet_file = open("poems/shakespeare_with_punc.txt", "r")
    sonnet = [line.strip() for line in sonnet_file.readlines()]
    shakespeare_sum = 0
    line_count = 0

    min = 100
    max = 0
    max_line = ""
    min_line = ""
    scores = []

    goodlinecount = 0
    goodlines = []

    for line in sonnet:
        temp = line.strip()
        score = s.gpt_2_score_line(temp)
        scores.append(score)
        if score > max:
            max = score
            max_line = line
        if score < min:
            min = score
            min_line = line
        shakespeare_sum += score
        line_count += 1
    avg_loss = shakespeare_sum/line_count
    for item in goodlines:
        print(item)

    #print(goodlinecount)
    print("mean: ", avg_loss)

    scores.sort()
    print("median: ", scores[int(len(scores)/2)])
    print("best line: ", min_line, " with a score of ", min)
    print("worst line: ", max_line, " with a score of ", max)

    #currently about 2.739 for bert on generated lines with no punc
    #shakespear score is 2.07 for bert, 6.8868 for gpt2 (for no punc), 6.23 for gpt-2 with punc
    """
    This is data generated from running GPT-2 on the sonnets with punctuation
    mean:  6.235340933724477
    median:  6.173692226409912
    best line:  If thou wilt leave me, do not leave me last,  with a score of  3.454342842102051
    worst line:  And captive good attending captain ill:  with a score of  10.742749214172363
    """

import sonnet_basic
import gpt_2_gen_bitdtfodb
import bert_verb
import sonnet_basic
import bulk_line_generator

if __name__ == '__main__':
    s = bert_verb.Scenery_Gen(model="gpt_2")
    #s = bert_verb.Scenery_Gen()
    generator = bulk_line_generator.Bulk_Gen()
    """
    generator = gpt_2_gen_bitdtfodb.gpt(None)
    sonnet = []
    for i in range(100):
        temp = generator.good_generation(None)
        sonnet.append(temp)
    """
    sonnet = generator.write_bulk_lines(1000, "death")

    #sonnet_file = open("poems/shakespeare_with_punc.txt", "r")
    #sonnet = [line.strip() for line in sonnet_file.readlines()]
    shakespeare_sum = 0
    line_count = 0

    min = 100
    max = 0
    max_line = ""
    min_line = ""
    scores = []

    goodlinecount = 0
    goodlines = []

    for i in range((int(len(sonnet)/2))):
        num = i * 2
        line = sonnet[num] + " " + sonnet[num+1]
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
        if score < 5.5:
            goodlines.append(line)
            goodlinecount += 1


    avg_loss = shakespeare_sum/line_count
    """
    for item in bertgood:
        print(item)
    for item in gpt2good:
        print(item)
    print("gpt2 thinks ", len(gpt2good), " are good")
    print("gpt2 thinks ", len(gpt2bad), " are bad")
    print("bert think ", len(bertgood), " are good")
    print("bert think ", len(bertbad), " are bad")
    print("they agree on ", len(goodlines))
    print(goodlinecount)
    """
    print("mean: ", avg_loss)

    scores.sort()
    print("median: ", scores[int(len(scores)/2)])
    print("best line: ", min_line, " with a score of ", min)
    print("worst line: ", max_line, " with a score of ", max)
    print("good lines, of which there are: ", goodlinecount)
    for item in goodlines:
        print(item)


    #currently about 2.739 for bert on generated lines with no punc
    #shakespear score is 2.07 for bert, 6.8868 for gpt2 (for no punc), 6.23 for gpt-2 with punc
    """
    This is data generated from running GPT-2 on the sonnets with punctuation
    mean:  6.235340933724477
    median:  6.173692226409912
    best line:  If thou wilt leave me, do not leave me last,  with a score of  3.454342842102051
    worst line:  And captive good attending captain ill:  with a score of  10.742749214172363
    If run on the sonnets in chunks of 4 lines, the loss goes down to 5.51
    
    If run on our sonnets in four lines, then mean is 7.15. Compare this to the score in isolation, at 7.6
    
    Other interesting numbers generating 1000 lines, with GPT-2s cutoff for a good line at 6, and a badline at 8, and Bert's cutoff for a good
    line at 1.25, and a bad one at 3.3:
    
    gpt2 thinks  52  are good
    gpt2 thinks  320  are bad
    bert think  81  are good
    bert think  175  are bad
    they agree on  3
    """

import sonnet_basic
import gpt_2_gen_bitdtfodb
import bert_verb
import sonnet_basic
import bulk_line_generator
import matplotlib
import matplotlib.pyplot as plt



def generate_and_score():
    s = bert_verb.Scenery_Gen(model="gpt_2")
    s2 = bert_verb.Scenery_Gen()
    # generator = bulk_line_generator.Bulk_Gen(model="gpt_2",templates_file="poems/paired_templates.txt", paired_templates=True)
    generator = bulk_line_generator.Bulk_Gen(model="gpt_2", templates_file="poems/ben_modified_templates.txt")
    """
       generator = gpt_2_gen_bitdtfodb.gpt(None)
       sonnet = []
       for i in range(100):
           temp = generator.good_generation(None)
           sonnet.append(temp)
       """
    #sonnet = generator.write_line_pairs_threshold(50, theme="trees") #takes a really long time to run
    #sonnet = generator.write_line_pairs(1000, theme="trees") #runs in a more reasonable amount of time
    sonnet = generator.write_bulk_lines(1000, theme="death")  # change number to change number of lines generated

    #sonnet_file = open("poems/poe.txt", "r")
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
        if score < 5.5:
            goodlines.append((line, score))
            goodlinecount += 1

    for i in range(len(sonnet)):
        print(sonnet[i], " with score ", scores[i])
    avg_loss = shakespeare_sum / line_count
    scores.sort()
    print("mean: ", avg_loss)
    print("median: ", scores[int(len(scores) / 2)])
    print("best line: ", min_line, " with a score of ", min)
    print("worst line: ", max_line, " with a score of ", max)
    print("good lines, of which there are: ", goodlinecount)
    for item in goodlines:
        print(item)

    # gpt_2_scores = []
    # bert_scores = []

    # start comment out here
    """
    for line in sonnet:
        temp = line.strip()
        score = s.gpt_2_score_line(temp)
        score2 = s2.score_line(temp)
    #end comment out here
        scores.append(score)
    """

    # gpt_2_scores.append(score)
    # bert_scores.append(score2)

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



def score_templates():
    s = bert_verb.Scenery_Gen(model="gpt_2")
    # generator = bulk_line_generator.Bulk_Gen(model="gpt_2",templates_file="poems/paired_templates.txt", paired_templates=True)
    generator = bulk_line_generator.Bulk_Gen(model="gpt_2", templates_file="poems/jordan_templates.txt")
    template_lines = generator.write_bulk_lines_with_template(10)
    template_scores = []
    print("finished generating, about to start scoring")
    for item in list(template_lines.keys()):
        scores = []
        for line in template_lines[item]:
            score = s.gpt_2_score_line(line)
            scores.append(score)
        if len(scores) == 0:
            temp_avg = -1
        else:
            temp_avg = sum(scores)/len(scores)
        template_scores.append((item, temp_avg))
    #template_scores.sort(key=lambda x:x[1], reverse=True)
    for obj in template_scores:
        print("the following template: ", obj[0], " had an average score of ", obj[1])
        for line in template_lines[obj[0]]:
            print(line)
        print()
        print()


if __name__ == '__main__':

    #generate_and_score()
    score_templates()

    """

    bert_avg = sum(bert_scores)/len(bert_scores)
    gpt_2_avg = sum(gpt_2_scores)/len(gpt_2_scores)

    print(bert_avg)
    print(gpt_2_avg)


    #plt.plot(gpt_2_scores, bert_scores, 'ro')
    plt.axis([3, 12, 0, 5])
    for i in range(len(sonnet)):
        plt.text(gpt_2_scores[i], bert_scores[i], sonnet[i], fontsize=10)
    plt.show()


    #currently about 2.739 for bert on generated lines with no punc
    #shakespear score is 2.07 for bert, 6.8868 for gpt2 (for no punc), 6.23 for gpt-2 with punc
    """

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

    """
    From initial testing of paired templates, with:
    
    NN VBD WITH NN AND JJ NNS QUITE VBN 0_1_0_1_0_10_1_0_1
    TO JJ NN AND VBG PRPO THERE. 0_10_10_1_01_0_1


    NN VBD AND VBD EVERY WHERE 01_01_0_10_10_1
    AND VB JJ NNS IN PRP$ NN POS NN. 0_1_0_10_1_0_10__1

    BUT AS THE NN SHOULD BY NN VB, 0_1_0_10_1_0_1_01
    THAT RB NN POS NN MIGHT RB VB 0_10_10__1_0_10_1
    
    mean:  7.491654950141907
    median:  7.4732136726379395
    best line:  convex seduced and huddled every where and tuck deep heaters in his skinner 's bob.  with a score of  6.201420783996582
    worst line:  warm packed with group and peaceful vines quite ripped to stricter janice and proving him there.  with a score of  9.093894958496094
    good lines, of which there are:  0
    
    Even when run on 10,000 lines, we still got no lines with a score less than 5.5.
    
    
    Shakespeare, scored in pairs
    mean:  5.909940609139431
    median:  5.87100887298584
    best line:  That thy unkindness lays upon my heart; Wound me not with thine eye, but with thy tongue:  with a score of  3.7415506839752197
    worst line:  And simple truth miscalled simplicity, And captive good attending captain ill:  with a score of  9.374080657958984
    good lines, of which there are:  318
    
    Random generation of our sonnets in two lines:
    
    mean:  7.496946856258361
    median:  7.429840087890625
    best line:  self is bequeathed, her whooping wry, warm of; untimely when her curtsy do remove,  with a score of  5.534509181976318
    worst line:  from quickest diggers, she deny deduct afraid admires would anytime compel  with a score of  9.99077033996582
    
    When I ran it on some lines from a neil gaiman blog:
    mean:  4.027967440454583
    median:  3.806979179382324
    best line:  He got the part, not because he was a legend, not because he was an icon, but because he was so good, and his interpretation of the character became, for me, definitive.  with a score of  2.7972352504730225
    worst line:  Eleven hours of drama.  with a score of  5.613171100616455
    
    From the brooke sonnet:
    mean:  6.294582435062954
    median:  6.028757095336914
    best line:  If I should die, think only this of me:  with a score of  4.515163898468018
    worst line:  Gives Somewhere back the thoughts by England given;  with a score of  9.089923858642578
    
    mean:  5.070588191350301
    median:  4.756784439086914
    best line:  Once upon a midnight dreary, while I pondered, weak and weary,  with a score of  4.196921348571777
    worst line:  Over many a quaint and curious volume of forgotten lore--  with a score of  6.516007900238037
    """
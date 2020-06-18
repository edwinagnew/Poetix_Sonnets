import bert_verb

if __name__ == '__main__':

    s = bert_verb.Scenery_Gen()
    shakespeare_sum = 0
    line_count = 0
    with open("poems/shakespeare_all_sonnets.txt") as shakespeare_sonnets:
        for line in shakespeare_sonnets:
            temp = line.strip()
            score = s.score_line(temp)
            shakespeare_sum += score
            line_count += 1
    avg_loss = shakespeare_sum/line_count #so far avg score is 2.0738
    print(avg_loss)


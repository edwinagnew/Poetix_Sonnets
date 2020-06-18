import sonnet_basic

import bert_verb
import sonnet_basic
import bulk_line_generator

if __name__ == '__main__':
    s = bert_verb.Scenery_Gen()
    generator = bulk_line_generator.Bulk_Gen()
    sonnet = generator.write_bulk_lines(1000, "death")
    shakespeare_sum = 0
    line_count = 0

    goodlinecount = 0

    for line in sonnet:
        temp = line.strip()
        score = s.score_line(temp)
        if score < 1.3:
            goodlinecount += 1
            print(line)
        shakespeare_sum += score
        line_count += 1
    avg_loss = shakespeare_sum/line_count
    print(goodlinecount)
    print(avg_loss) #currently about 2.739


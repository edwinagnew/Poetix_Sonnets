import scenery
import gpt_2

import random
import string

import pronouncing

s = scenery.Scenery_Gen(templates_file="poems/jordan_templates.txt")

gpt = gpt_2.gpt_gen(None, sonnet_method=s.get_pos_words, model="gpt2")

print("loaded")
def write_set(n_random, n_gpt, sonnet, gpt):
        sonnet.gender = random.choice([["i", "me", "my", "mine", "myself"], ["you", "your", "yours", "yourself"],  ["he", "him", "his", "himself"], ["she", "her", "hers", "herself"], ["we", "us", "our", "ours", "ourselves"], ["they", "them", "their", "theirs", "themselves"]])
        used_templates = []
        rhymes = []
        scores = []
        stanza = ""
        #for i in range(n_random):
        while len(stanza.split("\n")) <= n_random:
                r = None if len(rhymes) < 2 else rhymes[-2]
                template, meter = sonnet.get_next_template(used_templates, check_the_rhyme=r)
                line = sonnet.write_line_random(template, meter, r)
                if len(stanza) == 0 or stanza[-2] in ".?!":
                        line = line.capitalize()
                score = gpt.score_line(line)
                if score < 6:
                        stanza += line + "\n"
                        print("stanza now '" + stanza + "'")
                        rhymes.append(line.split()[-1])
                        used_templates.append(template)
                        scores.append(score)

        for j in range(n_gpt):
                r = None if len(rhymes) < 2 else rhymes[-2]
                template, meter = sonnet.get_next_template(used_templates, check_the_rhyme=r)
                line = gpt.good_generation(stanza, template=template, meter=meter, rhyme_word=r)
                line = line.replace(stanza.strip(), "").strip()
                #print(line, "\n=>", line.replace(stanza.strip(), ""), "\n=>", stanza)
                if len(stanza.split("\n")) == 0 or stanza[-2] in ".?!":
                        line = line.capitalize()
                stanza += line + "\n"
                rhymes.append(line.split()[-1])
                used_templates.append(template)
                score = gpt.score_line(line)
                scores.append(score)

        stanza = stanza.strip()
        total_score = gpt.score_line(stanza)

        return (stanza, total_score, scores)




s, t_s, sc = write_set(2,2,s,gpt)

print(s, t_s, sc)

#f = open("saved_objects/local_generated_stanzas/1rand_3g.txt", "a")

#f.write("1,3,\'" + s + "\'," + str(t_s) + "," + str(sc) + "\n")

#f.close()

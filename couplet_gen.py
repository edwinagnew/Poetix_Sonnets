import scenery
import gpt_2_gen

import random
import string

import pronouncing

s = scenery.Scenery_Gen(templates_file="poems/jordan_templates.txt")

gpt = gpt_2_gen.gpt(None, sonnet_method=s.get_pos_words, model="gpt2")

stanzas = []
for i in range(4):
        print(i)
        s.gender = random.choice([["i", "me", "my", "mine", "myself"], ["you", "your", "yours", "yourself"],  ["he", "him", "his", "himself"], ["she", "her", "hers", "herself"], ["we", "us", "our", "ours", "ourselves"], ["they", "them", "their", "theirs", "themselves"]])
        loss = 10
        while loss > 6:
                template, meter = random.choice(s.templates)
                if template[-1] == ">": template = template.replace("/.>", " ").replace("<", "")
                if template[-1] not in ",;" + string.ascii_uppercase: continue
                line = s.write_line(-1, template.split(), meter.split("_"))
                loss = gpt.score_line(line)
        #print(line, template, loss)
        next_template, next_meter = ",", "1"
        while next_template[-1] not in ".?":
                next_template, next_meter = random.choice(s.templates)
                if next_template[-1] == ">": next_template = next_template.split("<")[0] + "."
        #print("template 1: ", next_template)
        next_line = gpt.good_generation(line, template=template + " " + next_template, meter=meter + "_" + next_meter).replace(line, "")
        #print(line, "->", next_line)
        line += "\n" + next_line
        template += " " + next_template
        meter += "_" + next_meter


        next_template, next_meter = ".", "1"
        while next_template[-1] not in ",;" + string.ascii_uppercase or next_template.split()[0] in ["AND", "THAT", "OR", "SHALL", "WILL", "WHOSE"]:
                next_template, next_meter = random.choice(s.templates)
                if next_template[-1] == ">": next_template = next_template.replace("/.>", " ").replace("<", "")
        #print("template 2: ", next_template)
        rhyme = (line.split("\n")[-2].split()[-1])
        #print(rhyme)
        next_line = gpt.good_generation(line, template=template + " " + next_template, meter=meter + "_" + next_meter, rhyme_words=pronouncing.rhymes(rhyme)).replace(line, "")
        line += "\n" + next_line
        template += " " + next_template
        meter += "_" + next_meter


        next_template, next_meter = ",", "1"
        while next_template[-1] not in ".?":
                next_template, next_meter = random.choice(s.templates)
                if next_template[-1] == ">": next_template = next_template.split("<")[0] + "."
        #print("template 3: ", next_template)
        rhyme = (line.split("\n")[-2].split()[-1])
        next_line = gpt.good_generation(line, template=template + " " + next_template, meter=meter + "_" + next_meter, rhyme_words=pronouncing.rhymes(rhyme)).replace(line, "")
        line += "\n" + next_line

        stanzas.append((line, gpt.score_line([line, next_line])))
        #print(stanzas[-1])

print(stanzas)

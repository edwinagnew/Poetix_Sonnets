import scenery
import gpt_2_gen

import random
import string

s = scenery.Scenery_Gen(templates_file="poems/jordan_templates.txt")

gpt = gpt_2_gen.gpt(None, sonnet_method=s.get_pos_words)

couplets = []
for i in range(2):
        print(i)
        loss = 10
        while loss > 6:
                template, meter = random.choice(s.templates)
                if template[-1] == ">": template = template.replace("/.>"," ").replace("<", "")
                if template[-1] not in ",;" + string.ascii_lowercase: continue
                line = s.write_line(-1, template.split(), meter.split("_"))
                loss = gpt.score_line(line)
                print(line, loss)
        next_template, next_meter = ",", "1"
        while next_template[-1] not in ".?":
            next_template, next_meter = random.choice(s.templates)
            if next_template[-1] == ">": next_template = next_template.split("<")[0] + "."

        next_line = gpt.good_generation(line, template=template + " " + next_template, meter=meter + "_" + next_meter).replace(line, "")
        couplet = line + next_line
        couplets.append((couplet, gpt.score_line([line, next_line, couplet])))

print(couplets)
import scenery
import random

sc = scenery.Scenery_Gen()
gen_file = open("saved_objects/generated/poems_1-7.txt", "a")

theme_list = "love death peace war conflict nature beauty pain journey birth failure success tree ocean meadow flower".split()

for _ in range(10):
    theme = " ".join(random.sample(theme_list, 2))
    for k in [5,10]:
        t_lines = random.choice([0,0,0,1,2])
        for t_p in [True, False]:
            print(theme, k, t_lines, t_p)
            gen_file.write(theme + ", " + str(k) + ", " + str(t_lines) + ", " + str(t_p))
            try:
                poem = sc.write_poem_flex(theme=theme, k=k, verbose=False, theme_lines=t_lines, theme_progression=t_p)
                print(poem)
                gen_file.write("\t" + poem + "\n")

            except Exception as exc:
                print(exc)
                continue


gen_file.close()
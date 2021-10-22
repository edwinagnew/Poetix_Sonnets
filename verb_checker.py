# Verb checker baseline:
# For every verb, compare the score of that verb relative to choosing the best possible verb (ignoring meter etc) {use original GPT , and maybe a fine-tuned one}
# Create an initial corpus of our lines and then sort it by the magnitude of the difference with the best possible verb
# Create a classifier using “if there exists a verb that is X times better than word A, then its wrong”

# use sc.get_word_pos to find the verbs in each line

import poem_core
import scenery
s = scenery.Scenery_Gen()

file = open("verb_lines.txt", "r")
verb_lines = file.readlines()
verb_lines = [line[:-1] for line in verb_lines]
print(verb_lines)
print(list(s.all_templates_dict.keys()))
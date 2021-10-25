# Verb checker baseline:
# For every verb, compare the score of that verb relative to choosing the best possible verb (ignoring meter etc) {use original GPT , and maybe a fine-tuned one}
# Create an initial corpus of our lines and then sort it by the magnitude of the difference with the best possible verb
# Create a classifier using “if there exists a verb that is X times better than word A, then its wrong”

# use sc.get_word_pos to find the verbs in each line

from py_files import helper
import scenery
s = scenery.Scenery_Gen()
s.write_poem_revised(gpt_size="gpt2")

file = open("verb_lines.txt", "r")
verb_lines = file.readlines()
verb_lines = [line[:-1] for line in verb_lines]
# change later
verb_lines = verb_lines[0:14]

contains_verb = [line for line in verb_lines if len(s.get_template_from_line(line)) > 0]
contains_verb = [line for line in contains_verb if 'VB' in s.get_template_from_line(line)[0]]
print(contains_verb)

contains_verb_split = [helper.remove_punc(line) for line in contains_verb]
contains_verb_split = [line.split() for line in contains_verb_split]
print(contains_verb_split)

old_scores = [s.gpt.score_line(line) for line in contains_verb]
print(old_scores)

contains_verb_templates = [s.get_template_from_line(line)[0] for line in contains_verb]
print(contains_verb_templates)

contains_verb_templates_split = [helper.remove_punc(line) for line in contains_verb_templates]
contains_verb_templates_split = [line.split() for line in contains_verb_templates_split]
print(contains_verb_templates_split)

all_poss_verbs = []
for vb in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
    all_poss_verbs += s.get_pos_words(vb)
# change later
# all_poss_verbs = [all_poss_verbs[0]]
print(all_poss_verbs)

# And then for each sentence youll need to substitute every other possible verb, and then score the resulting sentence:
# new_scores = []
# for iter1 in range(len(contains_verb)):
#     # iter1 = contains_verb.index(line)
#     line = contains_verb[iter1]
#     verbs_in_template = [tag for tag in contains_verb_templates_split[iter1] if
#                          tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']]
#     verb_position = contains_verb_templates_split[iter1].index(verbs_in_template[0])
#     old_verb = line.split()[verb_position] #(do something to get the verb position)
#     new_scores.append([])
#     for iter2 in range(len(all_poss_verbs)):
#       # iter2 = all_poss_verbs.index(new_verb)
#       new_verb = all_poss_verbs[iter2]
#       new_line = line.replace(old_verb, new_verb)
#       new_score = s.gpt.score_line(new_line)
#       new_scores[iter1].append(new_score)

# only keep lowest (best) new score
new_scores = []
new_lines = []
for iter1 in range(len(contains_verb)):
    line = contains_verb[iter1]
    verbs_in_template = [tag for tag in contains_verb_templates_split[iter1] if
                         tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']]
    verb_position = contains_verb_templates_split[iter1].index(verbs_in_template[0])
    old_verb = line.split()[verb_position] #(do something to get the verb position)
    for iter2 in range(len(all_poss_verbs)):
      new_verb = all_poss_verbs[iter2]
      new_line = line.replace(old_verb, new_verb)
      new_score = s.gpt.score_line(new_line)
      if len(new_scores) < iter1 + 1:
          new_scores.append(new_score)
          new_lines.append(new_line)
      elif new_score < new_scores[iter1]:
          new_scores[iter1] = new_score
          new_lines[iter1] = new_line


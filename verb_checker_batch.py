# Verb checker baseline:
# For every verb, compare the score of that verb relative to choosing the best possible verb (ignoring meter etc) {use original GPT , and maybe a fine-tuned one}
# Create an initial corpus of our lines and then sort it by the magnitude of the difference with the best possible verb
# Create a classifier using “if there exists a verb that is X times better than word A, then its wrong”

# use sc.get_word_pos to find the verbs in each line

from py_files import helper
import scenery
import gpt_revised
import numpy as np

s = scenery.Scenery_Gen()
s.gpt = gpt_revised.gpt_gen(sonnet_object=s, model="custom fine_tuning/twice_retrained")

file = open("verb_lines.txt", "r")
verb_lines = file.readlines()
verb_lines = [line[:-1] for line in verb_lines]
# change later
#verb_lines = verb_lines[0:3]

contains_verb = [line for line in verb_lines if len(s.get_template_from_line(line)) > 0]
contains_verb = [line for line in contains_verb if 'VB' in s.get_template_from_line(line)[0]]
print(len(contains_verb))

contains_verb_split = [helper.remove_punc(line) for line in contains_verb]
contains_verb_split = [line.split() for line in contains_verb_split]
print(len(contains_verb_split))


#old_scores = [s.gpt.score_line(line) for line in contains_verb]
old_tokens = [s.gpt.tokenizer(c, return_tensors='pt')['input_ids'] for c in contains_verb]
old_scores = [s.gpt.score_line_new(toks) for toks in old_tokens]
print(len(old_scores))

contains_verb_templates = [s.get_template_from_line(line)[0] for line in contains_verb]
print(len(contains_verb_templates))

contains_verb_templates_split = [helper.remove_punc(line) for line in contains_verb_templates]
contains_verb_templates_split = [line.split() for line in contains_verb_templates_split]
print(len(contains_verb_templates_split))


all_poss_verbs = []
for vb in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
    all_poss_verbs += s.get_pos_words(vb)
# change later
# all_poss_verbs = [all_poss_verbs[0]]
all_poss_verbs = list(set(all_poss_verbs))
print(len(all_poss_verbs))

vb_poss = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

best_new_dict = {}

for i, line in enumerate(contains_verb):
    print(i)
    verbs_in_template = [tag for tag in contains_verb_templates_split[i] if
                         tag in vb_poss]
    if len(verbs_in_template) < 1:
        continue
    verb_position = contains_verb_templates_split[i].index(verbs_in_template[0])
    old_verb = line.split()[verb_position]
    new_lines = [line.replace(old_verb, new_verb) for new_verb in all_poss_verbs]
    new_line_toks = [s.gpt.tokenizer(n, return_tensors='pt')['input_ids'].to(s.gpt.model.device) for n in new_lines]
    new_line_scores = [s.gpt.score_tokens_new(toks) for toks in new_line_toks]

    top_idxs = list(np.argsort(new_line_scores))#[::-1]

    orig_idx = top_idxs.index(i)

    best_new_dict[line] = [(top_idxs.index(j), new_lines[j], new_line_scores[j]) for j in [orig_idx] + top_idxs[:3]]

    print(best_new_dict[line], "\n\n")

#out_str = ""
file = open("verb_scorer_results.txt", "a")

for line in best_new_dict:
    out_str = ""
    orig_pos, _, orig_score = best_new_dict[line][0]
    out_str += "\n\norig: " + line + ", pos=" + str(orig_pos) + ", score=" + np.round(orig_score, 4)

    for p2, l2, s2 in best_new_dict[line][1:]:
        out_str += "\ntop" + str(p2) + "," + l2 + ", score=" + np.round(s2, 4)

    file.write(out_str)


#file = open("verb_scorer_results.txt", "w")
#file.write(out_str)
file.close()

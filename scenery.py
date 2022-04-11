import random
import pickle
import numpy as np
import string
import pandas as pd

import helper

import theme_word_file

import gpt_revised
import word_embeddings
from difflib import SequenceMatcher

from nltk.corpus import wordnet as wn

import poem_core
import graph_reader


class Scenery_Gen(poem_core.Poem):
    def __init__(self, words_file="saved_objects/tagged_words.p",
                 syllables_file='saved_objects/cmudict-0.7b.txt',
                 extra_stress_file='saved_objects/edwins_extra_stresses.txt',
                 top_file='saved_objects/words/top_words.txt',
                 templates_file=('poems/templates_basic.txt', "poems/rhetorical_templates.txt"),
                 mistakes_file=None, tense=None):

        poem_core.Poem.__init__(self, words_file=words_file, templates_file=templates_file,
                                syllables_file=syllables_file, extra_stress_file=extra_stress_file, top_file=top_file,
                                mistakes_file=mistakes_file, tense=tense)
        self.vocab_orig = self.pos_to_words.copy()

        # with open('poems/kaggle_poem_dataset.csv', newline='') as csvfile:
        #   self.poems = csv.DictReader(csvfile)
        self.poems = list(pd.read_csv('poems/kaggle_poem_dataset.csv')['Content'])
        self.surrounding_words = {}

        # self.gender = random.choice([["he", "him", "his", "himself"], ["she", "her", "hers", "herself"]])

        self.theme_gen = theme_word_file.Theme()

        self.word_embeddings = word_embeddings.Sim_finder()

        self.theme = ""

        self.all_beam_histories = []

        self.beam_manager = None

        try:
            self.example_poems = pickle.load(open("saved_objects/saved_sample_poems.p", "rb"))
        except:
            self.example_poems = {}

        self.used_templates = []

    # override
    def get_pos_words(self, pos, meter=None, rhyme=None, phrase=()):
        """
        Gets all the words of a given POS
        Parameters
        ----------
        pos - the POS you want
        meter - (optional) returns only words which fit the given meter, e.g. 101
        phrase (optional) - returns only words which have a phrase in the dataset. in format ([word1, word2, word3], i) where i is the index of the word to change since the length can be 2 or 3
        """
        # similar/repeated word management
        if "*VB" in pos:
            ps = []
            for po in ["VB", "VBZ", "VBG", "VBD", "VBN", "VBP"]:
                ps += self.get_pos_words(po, meter=meter, rhyme=rhyme, phrase=phrase)
            return ps
        # if rhyme: return [w for w in self.get_pos_words(pos, meter=meter) if self.rhymes(w, rhyme)]
        if len(phrase) == 0 or len(phrase[0]) == 0:
            return super().get_pos_words(pos, meter=meter, rhyme=rhyme)
        else:
            if type(meter) == str: meter = [meter]
            ret = [word for word in self.pos_to_words[pos] if
                   word in self.dict_meters and any(m in self.dict_meters[word] for m in meter)]
            phrases = []
            for word in ret:
                phrase[0][phrase[1]] = word
                phrases.append(" ".join(phrase[0]))
            # print(phrases, ret)
            ret = [ret[i] for i in range(len(ret)) if self.phrase_in_poem_fast(phrases[i], include_syns=True)]
            return ret

    # @ovveride
    def suitable_last_word(self, word, line):
        pos = self.templates[line][0].split()[-1].split("sc")[-1]
        meter = self.templates[line][1].split("_")[-1]
        return pos in self.get_word_pos(word) and meter in self.dict_meters[word]

    def write_poem_flex(self, theme="love", verbose=False, random_templates=True, rhyme_lines=True, all_verbs=False,
                        theme_lines=0, k=5, alliteration=1, theme_threshold=0.5, no_meter=False,
                        theme_choice="or", theme_cutoff=0.35, sum_similarity=True,
                        theme_progression=False, story=False, story_file="saved_objects/story_graphs/love.txt",
                        gpt_size="gpt2", tense="present", internal_rhyme=0):
        if tense != self.tense:
            self.tense = tense
            if tense == None:
                tense = "basic"
            s = "poems/templates_" + tense + ".txt"
            with open(s) as tf:
                self.templates = [(" ".join(line.split()[:-1]), line.split()[-1]) for line in tf.readlines() if
                                  "#" not in line and len(line) > 1]
                if verbose: print("updated templates to ", s)
        if not self.gpt or gpt_size != self.gpt.model_size:
            if verbose: print("getting", gpt_size)
            self.gpt = gpt_revised.gpt_gen(sonnet_object=self, model=gpt_size)

        self.reset_gender()

        self.theme = theme

        if story:
            self.story_graph = graph_reader.Graph(txt_file=story_file)
            self.theme = (self.story_graph.curr.word, self.story_graph.curr.pos)
            theme = self.story_graph.curr.word
            self.story = [(self.theme[0], self.theme[1])]
            while len(self.story) < 4:
                self.story.append(self.story_graph.update())
            theme_progression = True
        else:
            self.story_graph = None

        if theme_lines > 0: self.update_theme_words(theme=theme)
        theme_contexts = self.theme_gen.get_cases(theme) if theme_lines > 0 else [""]
        if verbose and theme_lines: print("total lines", len(theme_contexts), "e.g.",
                                          random.sample(theme_contexts, min(len(theme_contexts), theme_lines)))

        if theme and not theme_progression:
            # sub_theme = " ".join([w for w in theme.split() if len(w) > 3])
            sub_theme = theme
            if not sub_theme: sub_theme = theme

            theme_words = {}
            theme_words[sub_theme] = {}

            for pos in ['NN', 'JJ', 'RB']:
                if pos not in theme_words[sub_theme]: theme_words[sub_theme][pos] = []

                if theme_choice == "and":
                    theme_words[sub_theme][pos] += self.get_diff_pos(sub_theme, pos, 10)
                else:
                    for t in sub_theme.split():
                        theme_words[sub_theme][pos] += self.get_diff_pos(t, pos, 10)
                if verbose: print("theme words, ", pos, ": ", len(theme_words[sub_theme][pos]),
                                  theme_words[sub_theme][pos])
            rhymes = []  # i think??
            if verbose: print("\n")
        else:
            rhymes = []
            theme_words = []
        # random.shuffle(rhymes)
        if theme and theme_progression:
            # sub_theme = " ".join([w for w in theme.split() if len(w) > 3])
            sub_theme = theme
            if self.story_graph == None:
                assert len(sub_theme.split()) == 2, sub_theme + "not good length"
                t1, t2 = sub_theme.split()
            else:
                t1, t2 = (None, None)
            stanza_words = {}
            stanza_themes = {}
            for stanza in range(4):  # first stanza only first theme, second and third both, last only second
                if t1:
                    stanza_theme = [t1 * int(stanza < 3), t2 * int(stanza > 0)]
                else:
                    stanza_theme = self.story[stanza][:-1]
                stanza_words[stanza] = self.vocab_orig.copy()
                for p in ["NN", "NNS", "ABNN"]:
                    stanza_words[stanza][p] = {word: s for (word, s) in self.pos_to_words[p].items() if
                                               self.word_embeddings.both_similarity(word, stanza_theme) > theme_cutoff}
                stanza_themes[stanza] = {}
                if not story:
                    stanza_theme = " ".join(stanza_theme).strip()
                for pos in ['NN', 'JJ', 'RB']:
                    stanza_themes[stanza][pos] = self.get_diff_pos(stanza_theme, pos, 10)
                    if verbose: print("stanza:", stanza, ", theme words, ", pos, ": ", len(stanza_themes[stanza][pos]),
                                      stanza_themes[stanza][pos])
                if self.story_graph != None and stanza > 0:
                    all_verbs = self.get_diff_pos(stanza_theme[0], "VB")
                    for pos in ["VB", "VBP", "VBZ", "VBG", "VBD", "VBN"]:
                        temp_set = set(self.pos_to_words[pos])
                        stanza_themes[stanza][pos] = [verb for verb in all_verbs if verb in temp_set]
                        if verbose:
                            print("theme words for verb ", stanza_theme[0], " for POS ", pos, " ",
                                  stanza_themes[stanza][pos])
                            print("number of verbs for the same category: ", len(stanza_themes[stanza][pos]), "\n")


        else:
            for p in ["NN", "NNS", "ABNN"]:
                if False and verbose: print("glove cutting", [w for w in self.pos_to_words[p] if
                                                    self.word_embeddings.ft_word_similarity(w,
                                                                                            self.theme.split()) > theme_cutoff > self.word_embeddings.gl_word_similarity(
                                                        w, self.theme.split())])
                if False and verbose: print("\n\nfasttext cutting", [w for w in self.pos_to_words[p] if
                                                           self.word_embeddings.ft_word_similarity(w,
                                                                                                   self.theme.split()) < theme_cutoff < self.word_embeddings.gl_word_similarity(
                                                               w, self.theme.split())])

                self.pos_to_words[p] = {word: s for (word, s) in self.pos_to_words[p].items() if
                                        self.word_embeddings.both_similarity(word, self.theme.split()) > theme_cutoff}
                if verbose and False: print("ended for", p, len(self.vocab_orig[p]), len(self.pos_to_words[p]),
                                  set(self.pos_to_words[p]))
        self.set_meter_pos_dict()

        samples = ["\n".join(random.sample(theme_contexts, theme_lines)) if theme_lines else "" for i in
                   range(4)]  # one for each stanza
        if verbose: print("samples, ", samples)
        # rhymes = []
        # theme = None

        lines = []
        used_templates = []
        choices = []

        internal_rhymes = []
        # first three stanzas

        self.gpt_past = ""
        line_number = 0
        while line_number < 14:
            if line_number % 4 == 0:
                if verbose: print("\n\nwriting stanza", 1 + line_number / 4)
                # else:
                #    if line_number > 0: print("done")
                #    if len(choices) == 0: print("\nwriting stanza", 1 + line_number/4, end=" ...")
                alliterate = alliteration
                if theme_progression:
                    self.words_to_pos = stanza_words[int(line_number / 4)]
                    self.set_meter_pos_dict()
            lines = lines[:line_number]
            used_templates = used_templates[:line_number]
            self.all_beam_histories = self.all_beam_histories[:line_number]

            if internal_rhyme > 0:
                internal_rhymes = " ".join(lines[-min(len(lines), internal_rhyme):]).lower().split()
                if verbose: print("words before the internal rhyme are as follows", internal_rhymes)

            if rhyme_lines and line_number % 4 >= 2:
                r = helper.remove_punc(lines[line_number - 2].split()[-1])  # last word in rhyming couplet
            elif rhyme_lines and line_number == 13:
                r = helper.remove_punc(lines[12].split()[-1])
            elif rhyme_lines and theme:
                # r = "__" + random.choice(rhymes)
                r = None  # r = set(rhymes)
            else:
                r = None

            if random_templates:
                template, meter = self.get_next_template(used_templates, end=r)
                if not template:
                    if verbose: print("didnt work out for", used_templates, r)
                    continue
            else:
                template, meter = self.templates[line_number]

            if no_meter:
                meter = {}

            # if r and len()
            alliterating = "_" not in template and alliterate and random.random() < 0.5  # 0.3
            if alliterating:
                if random.random() < 0.85:
                    letters = string.ascii_lowercase
                else:
                    letters = "s"
                    # letters = string.ascii_lowercase
            else:
                letters = None

            # self.gpt_past = str(theme_lines and theme.upper() + "\n") + "\n".join(lines) #bit weird but begins with prompt if trying to be themey
            # self.gpt_past = " ".join(theme_words) + "\n" + "\n".join(lines)
            self.gpt_past = samples[0] + "\n"
            for i in range(len(lines)):
                if i % 4 == 0: self.gpt_past += samples[i // 4] + "\n"
                self.gpt_past += lines[i] + "\n"
            self.reset_letter_words()
            if verbose:
                print("\nwriting line", line_number)
                print("alliterating", alliterating, letters)
                print(template, meter, r)
            t_w = theme_words[sub_theme] if not theme_progression else stanza_themes[line_number // 4]

            line = self.write_line_gpt(template, meter, rhyme_word=r, flex_meter=True, verbose=verbose,
                                       all_verbs=all_verbs, alliteration=letters, theme_words=t_w,
                                       theme_threshold=theme_threshold, internal_rhymes=internal_rhymes)
            # create new Line_Generator object

            if line: line_arr = line.split()
            if line and rhyme_lines and not random_templates and line_number % 4 < 2:
                rhyme_pos = self.templates[min(line_number + 2, 13)][0].split()[-1]
                # if any(self.rhymes(line.split()[-1], w) for w in self.get_pos_words(rhyme_pos)):
                if len(self.get_pos_words(rhyme_pos, rhyme=line.split()[-1])) > 0.001 * len(
                        self.get_pos_words(rhyme_pos)):
                    if "a" in line_arr and line_arr[line_arr.index("a") + 1][0] in "aeiou": line = line.replace("a ",
                                                                                                                "an ")
                    if len(lines) % 4 == 0 or any(p in lines[-1][-2:] for p in ".?!"): line = line.capitalize()
                    if verbose: print("wrote line which rhymes with", rhyme_pos, ":", line)
                    # score = self.gpt.score_line("\n".join(random.sample(theme_contexts, min(len(theme_contexts), theme_lines))) + line)
                    score = self.gpt.score_line(line)
                    choices.append(
                        (score, line, template))  # scores with similarity to a random other line talking about it
                    if len(choices) == k:
                        best = min(choices)
                        if verbose: print("out of", len(choices), "chose", best)
                        lines.append(best[1])
                        used_templates.append(best[2])
                        line_number += 1
                        choices = []
                        if best[3]: alliterate -= 1
                else:
                    if verbose: print(line_number, "probably wasnt going to get a rhyme with", rhyme_pos)
                    # self.pos_to_words[template.split()[-1]][line.split()[-1]] /= 2
            elif line:
                if "a" in line_arr and line_arr[line_arr.index("a") + 1][0] in "aeiou": line = line.replace("a ", "an ")
                if len(lines) % 4 == 0 or any(p in lines[-1][-2:] for p in ".?!"): line = line.capitalize()
                line = line.replace(" i ", " I ").replace("\ni", "\nI")
                if verbose: print("wrote line", line)
                if len(lines) % 4 == 0:
                    samp = theme + "\n" + samples[len(lines) // 4] + "\n" + line
                    choices.append((self.gpt.score_line(samp), line, template, alliterating))
                else:
                    curr_stanza = "\n".join(lines[len(lines) - (len(lines) % 4):])
                    # line_score = self.gpt.score_line(theme + "\n" + curr_stanza + "\n" + line)
                    line_score = self.gpt.score_line(curr_stanza + "\n" + line)
                    if sum_similarity: line_score *= sum(
                        [self.word_embeddings.ft_word_similarity(w, theme.split()) for w in line.split() if
                         "NN" in self.get_word_pos(w) or "JJ" in self.get_word_pos(w)])
                    choices.append((line_score, line, template, alliterating))
                if len(choices) == k:
                    best = min(choices)
                    if verbose:
                        print(choices)
                        print(line_number, ":out of", len(choices), "chose", best)
                    lines.append(best[1])
                    used_templates.append(best[2])
                    line_number += 1
                    choices = []
                    if best[3]: alliterate -= 1
                    last = helper.remove_punc(lines[-1].split()[-1])
                    if last in rhymes: rhymes = [r for r in rhymes if r != last]
            else:
                if verbose: print("no line", template, r)
                if random.random() < (1 / len(self.templates) * 2) * (1 / k):
                    if verbose: print("so resetting randomly")
                    if line_number == 13:
                        line_number = 12
                    else:
                        line_number -= 2

        # if not verbose and len(choices) == 0: print("done")
        ret = ("         ---" + theme.upper() + "---       , k=" + str(k) + "\n") if theme else ""
        for cand in range(len(lines)):
            ret += str(lines[cand]) + "\n"
            if (cand + 1) % 4 == 0: ret += "\n"
        if verbose: print(ret)

        self.pos_to_words = self.vocab_orig.copy()

        return ret

    def write_poem_revised(self, theme="love", verbose=False, rhyme_lines=True, all_verbs=False,
                           theme_lines=0, k=1, alliteration=1, theme_threshold=0.5, no_meter=False,
                           theme_choice="or", theme_cutoff=0.35, sum_similarity=True, weight_repetition=0,
                           theme_progression=False, story=False, story_file="saved_objects/story_graphs/love.txt",
                           gpt_size="custom fine_tuning/twice_retrained", tense="rand", internal_rhyme=1, dynamik=False,
                           random_word_selection=False, verb_swap=False, rep_penalty=1,
                           b=3, b_inc=1, beam_score="token", phi_score=False):

        # deleted random_templates
        # changed default k to 1, for testing efficiency purposes

        # test all_verbs
        # test theme_lines - either int (0 or positive int) or str (poem or stanza)
        # test dif combinations of theme words' pos for theme_choice (for 2 or more word themes)
        # test theme_cutoff and theme_progression
        # no_meter and rhyme_lines
        # see if sum_similarity is used in write_poem_revised
        # test dif numbers for weight_rep
        # ignore story and story_file b_inc phi_score
        # gpt_size twice_retrained vs xl
        # see if some themes are better in past vs present
        # test random_word_selection for low beam
        # compare token and line for beam_score

        if tense == "rand": tense = random.choice(["present", "past"])
        if tense != self.tense:
            self.tense = tense
            if tense == None:
                tense = "basic"
            s = "poems/templates_" + tense + ".txt"
            self.templates = self.get_templates_from_file(s)
            if verbose: print("updated templates to ", s)
        if not self.gpt or gpt_size != self.gpt.model_size:
            if verbose: print("getting", gpt_size)
            self.gpt = gpt_revised.gpt_gen(sonnet_object=self, model=gpt_size)

            if b > 1:
                self.beam_manager = gpt_revised.BeamManager(gpt_size, self.gpt.tokenizer, sonnet_object=self,
                                                            verbose=verbose, weight_repetition=weight_repetition, rep_penalty=rep_penalty)

        self.reset_gender()

        self.all_beam_histories = []

        self.theme = theme

        if story:
            self.story_graph = graph_reader.Graph(txt_file=story_file)
            self.theme = (self.story_graph.curr.word, self.story_graph.curr.pos)
            theme = self.story_graph.curr.word
            self.story = [(self.theme[0], self.theme[1])]
            while len(self.story) < 4:
                self.story.append(self.story_graph.update())
            theme_progression = True
        else:
            self.story_graph = None

        if type(theme_lines) == int:
            if theme_lines > 0: self.update_theme_words(theme=theme)
            theme_contexts = self.theme_gen.get_cases(theme) if theme_lines > 0 else [""]
            if verbose and theme_lines: print("total lines", len(theme_contexts), "e.g.",
                                              random.sample(theme_contexts, min(len(theme_contexts), theme_lines)))
            sample_seed = "\n".join(random.sample(theme_contexts, theme_lines)) if theme_lines else ""
            # L- set sample_seed if theme_lines > 0, otherwise set sample_seed to an empty string
        else:
            assert theme_lines in ["stanza", "poem"], "expected 'stanza' or 'poem'"
            if theme not in self.example_poems:
                if verbose: print("generating seed poem first")
                seed_poem = self.write_poem_revised(theme=theme, verbose=verbose, rhyme_lines=False,
                                                    all_verbs=all_verbs,
                                                    theme_lines=0, k=1, b=1, alliteration=0,
                                                    theme_threshold=theme_threshold,
                                                    no_meter=no_meter, theme_choice=theme_choice,
                                                    theme_cutoff=theme_cutoff,
                                                    sum_similarity=sum_similarity, weight_repetition=False,
                                                    theme_progression=theme_progression, story=story,
                                                    story_file=story_file, rep_penalty=rep_penalty,
                                                    gpt_size=gpt_size, tense=tense, internal_rhyme=0, dynamik=False,
                                                    random_word_selection=random_word_selection)

                #self.example_poems[theme] = "\n".join(seed_poem.split("\n")[1:])
                self.example_poems[theme] = seed_poem.split("-")[-1].strip()
                pickle.dump(self.example_poems, open("saved_objects/saved_sample_poems.p", "wb"))

            sample_lines = self.example_poems[theme].split("\n")
            if theme_lines == "poem":
                sample_seed = "\n".join(sample_lines)
                # L- seed poem with saved poem for that theme
            elif theme_lines == "stanza":
                sample_seed = "\n".join(sample_lines[10:14])  # last 4 lines
                # L- seed poem with last 4 lines of saved poem for that theme (change to last 2 lines? 6 lines?)

        if verbose: print("samples: ", sample_seed)

        if theme and not theme_progression:
            # sub_theme = " ".join([w for w in theme.split() if len(w) > 3])
            sub_theme = theme
            if not sub_theme: sub_theme = theme

            theme_words = {}
            theme_words[sub_theme] = {}

            for pos in ['NN', 'JJ', 'RB']:
                if pos not in theme_words[sub_theme]: theme_words[sub_theme][pos] = []

                if theme_choice == "and":
                    theme_words[sub_theme][pos] += self.get_diff_pos(sub_theme, pos, 10)
                else:
                    for t in sub_theme.split():
                        theme_words[sub_theme][pos] += self.get_diff_pos(t, pos, 10)
                if verbose: print("theme words, ", pos, ": ", len(theme_words[sub_theme][pos]),
                                  theme_words[sub_theme][pos])
            rhymes = []  # i think??
            if verbose: print("\n")
        else:
            rhymes = []
            theme_words = []
        # random.shuffle(rhymes)
        if theme and theme_progression:
            # sub_theme = " ".join([w for w in theme.split() if len(w) > 3])
            sub_theme = theme
            if self.story_graph == None:
                assert len(sub_theme.split()) == 2, sub_theme + "not good length"
                t1, t2 = sub_theme.split()
            else:
                t1, t2 = (None, None)
            stanza_words = {}
            stanza_themes = {}
            for stanza in range(4):  # first stanza only first theme, second and third both, last only second
                if t1:
                    stanza_theme = [t1 * int(stanza < 3), t2 * int(stanza > 0)]
                else:
                    stanza_theme = self.story[stanza][:-1]
                stanza_words[stanza] = self.vocab_orig.copy()
                for p in ["NN", "NNS", "ABNN"]:
                    stanza_words[stanza][p] = {word: s for (word, s) in self.pos_to_words[p].items() if
                                               self.word_embeddings.both_similarity(word, stanza_theme) > theme_cutoff}
                stanza_themes[stanza] = {}
                if not story:
                    stanza_theme = " ".join(stanza_theme).strip()
                for pos in ['NN', 'JJ', 'RB']:
                    stanza_themes[stanza][pos] = self.get_diff_pos(stanza_theme, pos, 10)
                    if verbose: print("stanza:", stanza, ", theme words, ", pos, ": ", len(stanza_themes[stanza][pos]),
                                      stanza_themes[stanza][pos])
                if self.story_graph != None and stanza > 0:
                    all_verbs = self.get_diff_pos(stanza_theme[0], "VB")
                    for pos in ["VB", "VBP", "VBZ", "VBG", "VBD", "VBN"]:
                        temp_set = set(self.pos_to_words[pos])
                        stanza_themes[stanza][pos] = [verb for verb in all_verbs if verb in temp_set]
                        if verbose:
                            print("theme words for verb ", stanza_theme[0], " for POS ", pos, " ",
                                  stanza_themes[stanza][pos])
                            print("number of verbs for the same category: ", len(stanza_themes[stanza][pos]), "\n")

        else:
            for p in ["NN", "NNS", "ABNN"]:
                if False and verbose: print("glove cutting", [w for w in self.pos_to_words[p] if
                                                    self.word_embeddings.ft_word_similarity(w,
                                                                                            self.theme.split()) > theme_cutoff > self.word_embeddings.gl_word_similarity(
                                                        w, self.theme.split())])
                if False and verbose: print("\n\nfasttext cutting", [w for w in self.pos_to_words[p] if
                                                           self.word_embeddings.ft_word_similarity(w,
                                                                                                   self.theme.split()) < theme_cutoff < self.word_embeddings.gl_word_similarity(
                                                               w, self.theme.split())])

                self.pos_to_words[p] = {word: s for (word, s) in self.pos_to_words[p].items() if
                                        self.word_embeddings.both_similarity(word, self.theme.split()) > theme_cutoff}
                if verbose and False: print("ended for", p, len(self.vocab_orig[p]), len(self.pos_to_words[p]),
                                  set(self.pos_to_words[p]))
        self.set_meter_pos_dict()

        # rhymes = []
        # theme = None

        n_regened = []

        lines = []
        used_templates = []
        choices = []

        internal_rhymes = []
        # first three stanzas

        self.gpt_past = ""
        line_number = 0
        while line_number < 14:
            if theme_lines and type(theme_lines) == int:
                sample_seed = "\n".join(random.sample(theme_contexts, theme_lines))
            if line_number % 4 == 0:
                self.prev_rhyme = None
                if verbose: print("\n\nwriting stanza", 1 + line_number / 4)
                # else:
                #    if line_number > 0: print("done")
                #    if len(choices) == 0: print("\nwriting stanza", 1 + line_number/4, end=" ...")
                alliterate = alliteration
                if theme_progression:
                    self.words_to_pos = stanza_words[int(line_number / 4)]
                    self.set_meter_pos_dict()
            lines = lines[:line_number]
            used_templates = used_templates[:line_number]
            self.all_beam_histories = self.all_beam_histories[:line_number]

            if internal_rhyme > 0:
                internal_rhymes = " ".join(lines[-min(len(lines), internal_rhyme):]).lower().split()
                if verbose: print("words before the internal rhyme are as follows", internal_rhymes)

            if rhyme_lines and line_number % 4 >= 2:
                r = helper.remove_punc(lines[line_number - 2].split()[-1])  # last word in rhyming couplet
                self.prev_rhyme = helper.remove_punc(lines[-1].split()[-1])
            elif rhyme_lines and line_number == 13:
                r = helper.remove_punc(lines[12].split()[-1])
            else:
                r = None

            templates = []
            meters = []

            if no_meter:
                template, _ = self.get_next_template(used_templates, end=r)
                templates.append(template)
            else:
                if verbose: print("\nlooking for the next template to rhyme with '" + str(r) + "'", used_templates,
                                  " :")
                for _ in range(k):
                    tries = 0
                    meter_dict = None
                    template = ""
                    while tries < len(self.templates) and meter_dict is None:
                        template, meter = self.get_next_template(used_templates, end=r)
                        if not template:
                            print("no template", 1 / 0)

                        meter_dict = self.get_meter_dict(template, meter, rhyme_word=r, verbose=verbose)
                        tries += 1

                    if template and meter_dict:
                        templates.append(template)
                        meters.append(meter_dict)

                if not any(m for m in meters) or len(meters) == 0:
                    if verbose: print("no meters resetting randomly")
                    if line_number == 13:
                        line_number = 12
                    else:
                        line_number -= 2

                    continue
                else:
                    if verbose: print('success')

            alliterating = alliterate and random.random() <= alliteration / 4  # 0.3
            if alliterating:
                alliterate -= 1
                if random.random() < 0.85:
                    letters = string.ascii_lowercase
                else:
                    letters = "s"
                    # letters = string.ascii_lowercase
            else:
                letters = None

            self.gpt_past = sample_seed
            if len(self.gpt_past) > 0 and self.gpt_past[-1] != "\n": self.gpt_past += "\n"
            for i in range(len(lines)):
                # if i % 4 == 0: self.gpt_past += samples[i // 4] + "\n"
                self.gpt_past += lines[i]  # + "\n"
            self.reset_letter_words()

            if verbose:
                print("\nwriting line", line_number)
                print("alliterating", alliterating, letters)
                print(templates, r)
            t_w = theme_words[sub_theme] if not theme_progression else stanza_themes[line_number // 4]

            if b <= 1:
                self.line_gen = gpt_revised.Line_Generator(self, self.gpt, templates, meters, rhyme_word=r,
                                                           theme_words=t_w,
                                                           alliteration=letters, weight_repetition=weight_repetition,
                                                           prev_lines=self.gpt_past, internal_rhymes=internal_rhymes,
                                                           verbose=verbose, branching=b, b_inc=b_inc,
                                                           phi_score=phi_score,
                                                           random_selection=random_word_selection,
                                                           beam_score=beam_score)
                # all_beams = self.line_gen.complete_lines()
                completed_beams = self.line_gen.beam_search_tokenwise()

                self.all_beam_histories.append(self.line_gen.beam_history)
            else:
                completed_beams = {}
                # new code
                self.beam_manager.reset_to_new_line(rhyme_word=r, theme_words=t_w, alliteration=letters,
                                                    seed=self.gpt_past, internal_rhymes=internal_rhymes)

                for i in range(len(templates)):
                    self.beam_manager.partial_lines = []
                    outs = self.beam_manager.generate(templates[i], meters[i], b)
                    completed_beams[templates[i]] = outs

            best = (np.inf, "", "", 0)

            # for t in all_beams:
            #    for p_l in all_beams[t]:
            #        line = p_l.curr_line
            for t in completed_beams:
                for line in completed_beams[t]:
                    # line = line.replace("[EOL]", "\n")
                    if len(lines) % 4 == 0 or any(p in lines[-1][-2:] for p in ".?!"): line = line.capitalize()
                    line = line.replace(" i ", " I ").replace("\ni", "\nI")
                    line = helper.fix_caps(line)

                    # check to see whether line similarity is too bad
                    similarities = [len(set.intersection(set(old_line.lower().split()), set(line.lower().split()))) for old_line in lines]
                    #assert len(line.split()) == len(t.split()), (line, t)
                    if len(line.split()) != len(t.replace(" POS ", " ").split()) or len(similarities) > 0 and max(similarities)/len(line.split()) > 0.5: # if the new line is at least half as similar as any previous one, ignore it
                        continue

                    if line[-1] != "\n": line += "\n"
                    cand_poem = "".join(lines) + line
                    inputs = self.gpt.tokenizer(cand_poem, return_tensors="pt")
                    best = min(best, (self.gpt.score_tokens_new(inputs['input_ids'].to(self.gpt.model.device)), line, t,
                                      inputs['input_ids'].size(1)))

            if best[0] == np.inf:
                if verbose: print("failed")
                success = False
                line_score = np.inf
            else:
                success = True
                line_score = self.gpt.score_line(best[1])  # /len(best[1].split())
                # line_score = best[0] * len("".join(lines + [best[1]]))
                if verbose: print("the best was", line_score, best)

            # bound = 5.8 if "custom" in gpt_size else 6
            bound = 5.5
            if (line_score > bound and dynamik) or not success:
                if verb_swap and success:
                    new_line, new_score = self.swap_verbs(best[1], best[2], r, seed="".join(lines), verbose=verbose)
                    if new_score <= bound:
                        lines.append(new_line)
                        used_templates.append(best[2])
                        line_number += 1
                n_regened.append(line_number)
                if n_regened.count(line_number) > 5: # restart stanza if too many fails
                    if verbose: print("failed", n_regened, ", resetting to beginning of stanza")
                    line_number = 4 * (line_number//4)
                if verbose and (not verb_swap or new_score > bound):
                    print("best option not up to snuff, trying again.")
                    print(best[1] + " was the best for line", line_number)
            else:
                lines.append(best[1])
                used_templates.append(best[2])
                line_number += 1

                last = helper.remove_punc(lines[-1].split()[-1])
                if last in rhymes: rhymes = [r for r in rhymes if r != last]

        self.used_templates = used_templates
        all_templates = [helper.remove_punc(t[0]) for t in self.templates]
        template_indices = [all_templates.index(helper.remove_punc(t)) if helper.remove_punc(t) in all_templates else -1 for t in used_templates]
        # if not verbose and len(choices) == 0: print("done")
        ret = ("         ---" + theme.upper() + "---       , k=" + str(k) + ", b=" + str(b) + ", rep=" + str(rep_penalty) + "\n") if theme else ""
        for cand in range(len(lines)):
            ret += "(" + str(template_indices[cand]) + ")\t" + str(lines[cand])
            if (cand + 1) % 4 == 0: ret += "\n"
        if verbose: print(ret)
        if verbose and dynamik: print("regened", n_regened)

        self.pos_to_words = self.vocab_orig.copy()

        if self.save_poems:
            self.saved_poems[theme] = "\n".join(ret.split("\n")[1:])  # to cut out title

        return ret

    def update_theme_words(self, word_dict={}, theme=None):
        if theme: word_dict = self.theme_gen.get_theme_words(theme)
        for pos in word_dict:
            self.pos_to_words["sc" + pos] = word_dict[pos]

    def phrase_in_poem_fast(self, words, include_syns=False):
        if type(words) == list:
            if len(words) <= 1: return True
        else:
            words = words.split()
        if len(words) > 2: return self.phrase_in_poem_fast(words[:2],
                                                           include_syns=include_syns) and self.phrase_in_poem_fast(
            words[1:], include_syns=include_syns)
        if words[0] == words[1]: return True
        if words[0][-1] in ",.?;>" or words[1][-1] in ",.?;>": return self.phrase_in_poem_fast(
            (words[0] + " " + words[1]).translate(str.maketrans('', '', string.punctuation)), include_syns=include_syns)
        if words[0] in self.gender: return True  # ?
        # words = " "+ words + " "

        # print("evaluating", words)

        if include_syns:
            syns = []
            for j in words:
                syns.append([l.name() for s in wn.synsets(j) for l in s.lemmas() if l.name() in self.dict_meters])
            contenders = set(words[0] + " " + w for w in syns[1])
            contenders.update([w + " " + words[1] for w in syns[0]])
            # print(words, ": " , contenders)
            return any(self.phrase_in_poem_fast(c) for c in contenders)

        if words[0] in self.surrounding_words:
            return words[1] in self.surrounding_words[words[0]]
        elif words[1] in self.surrounding_words:
            return words[0] in self.surrounding_words[words[1]]
        else:
            self.surrounding_words[words[0]] = set()
            self.surrounding_words[words[1]] = set()
            translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
            for poem in self.poems:
                poem = " " + poem.lower() + " "
                for word in words:
                    a = poem.find(word)
                    if a != -1 and poem[a - 1] not in string.ascii_lowercase and poem[
                        a + len(word)] not in string.ascii_lowercase:
                        # print(a, poem[a:a+len(word)])
                        p_words = poem.translate(translator).split()  # remove punctuation and split
                        if word in p_words:  # not ideal but eg a line which ends with a '-' confuses it
                            a = p_words.index(word)
                            if a - 1 >= 0 and a - 1 < len(p_words): self.surrounding_words[word].add(p_words[a - 1])
                            if a + 1 >= 1 and a + 1 < len(p_words): self.surrounding_words[word].add(p_words[a + 1])
            return self.phrase_in_poem_fast(words, include_syns=False)

    def close_adv(self, input, num=5, model_topn=50):
        if type(input) == str:
            positive = input.split() + ['happily']
        else:
            positive = list(input) + ["happily"]
        negative = ['happy']
        all_similar = self.word_embeddings.fasttext_model.most_similar(positive, negative, topn=model_topn)

        def score(candidate):
            ratio = SequenceMatcher(None, candidate, input).ratio()
            looks_like_adv = 1.0 if candidate.endswith('ly') else 0.0
            return ratio + looks_like_adv

        close = sorted([(word, score(word)) for word, _ in all_similar], key=lambda x: -x[1])
        return [word[0] for word in close[:num] if word[0] in self.pos_to_words["RB"]]

    def close_jj(self, input, num=5, model_topn=50):
        negative = ['darkness']
        if type(input) == str:
            positive = input.split() + ['dark']
        else:
            positive = list(input) + ["dark"]
        all_similar = self.word_embeddings.fasttext_model.most_similar(positive, negative, topn=model_topn)
        close = [word[0] for word in all_similar if word[0] in self.pos_to_words["JJ"]]

        return close

    def close_nn(self, input, num=5, model_topn=50):
        negative = ['dark']
        if type(input) == str:
            positive = input.split() + ['darkness']
        else:
            positive = list(input) + ["darkness"]
        all_similar = self.word_embeddings.fasttext_model.most_similar(positive, negative, topn=model_topn)
        close = [word[0] for word in all_similar if
                 word[0] in self.pos_to_words["NN"] or word[0] in self.pos_to_words["NNS"] or word[0] in
                 self.pos_to_words['ABNN']]

        return close

    def close_vb(self, input, num=5, model_topn=75):
        positive = input
        all_similar = self.word_embeddings.fasttext_model.most_similar(positive, topn=model_topn)
        close = [word[0] for word in all_similar if
                 word[0] in self.pos_to_words["VB"] or word[0] in self.pos_to_words["VBP"] or
                 word[0] in self.pos_to_words["VBD"] or word[0] in self.pos_to_words["VBN"] or
                 word[0] in self.pos_to_words["VBZ"] or word[0] in self.pos_to_words["VBG"]]

        return close

    def get_diff_pos(self, word, desired_pos, n=10):
        closest_words = [noun for noun in self.word_embeddings.get_close_words(word) if
                         (noun in self.pos_to_words["NN"] or noun in self.pos_to_words["NNS"])]
        if desired_pos == "JJ":
            index = 0
            words = set(self.close_jj(word))
            while len(words) < n and index < min(5, len(closest_words)):
                words.update(self.close_jj(closest_words[index]))
                index += 1
            return list(words)

        if desired_pos == "RB":
            index = 0
            words = set(self.close_adv(word))
            while len(words) < n and index < min(5, len(closest_words)):
                words.update(self.close_adv(closest_words[index]))
                index += 1
            return list(words)

        if "NN" in desired_pos:
            index = 0
            words = set(self.close_nn(word))
            while len(words) < n and index < min(5, len(closest_words)):
                words.update(self.close_nn(closest_words[index]))
                index += 1
            return list(words)

        if "VB" in desired_pos:
            index = 0
            words = set(self.close_vb(word))
            while (len(words) < n and index < 5):
                words.update(self.close_vb(closest_words[index]))
                index += 1
            return list(words)

        return [w for w in closest_words if desired_pos in self.get_word_pos(w)]

    def write_poem_no_template(self, theme="love", verbose=False, rhyme_lines=False, k=1, alliteration=True,
                               theme_threshold=0.5, no_meter=False, theme_cutoff=0.35, sum_similarity=True,
                               gpt_size="gpt2",
                               regen_threshhold=6, theme_lines=0, ):

        if not self.gpt or gpt_size != self.gpt.model_size:
            if verbose: print("getting", gpt_size)
            self.gpt = gpt_revised.gpt_gen(sonnet_object=self, model=gpt_size)

        self.reset_gender()

        self.theme = theme

        if theme_lines > 0: self.update_theme_words(theme=theme)
        theme_contexts = self.theme_gen.get_cases(theme) if theme_lines > 0 else [""]
        if verbose and theme_lines: print("total lines", len(theme_contexts), "e.g.",
                                          random.sample(theme_contexts, min(len(theme_contexts), theme_lines)))

        if theme:
            # sub_theme = " ".join([w for w in theme.split() if len(w) > 3])
            sub_theme = theme
            if not sub_theme: sub_theme = theme

            theme_words = {}
            theme_words[sub_theme] = {}

            for pos in ['NN', 'JJ', 'RB']:
                if pos not in theme_words[sub_theme]: theme_words[sub_theme][pos] = []

                for t in sub_theme.split():
                    theme_words[sub_theme][pos] += self.get_diff_pos(t, pos, 10)
                if verbose: print("theme words, ", pos, ": ", len(theme_words[sub_theme][pos]),
                                  theme_words[sub_theme][pos])
            rhymes = []  # i think??
            if verbose: print("\n")
        else:
            rhymes = []
            theme_words = []
        # random.shuffle(rhymes)
        for p in ["NN", "NNS", "ABNN"]:
            if False and verbose: print("glove cutting", [w for w in self.pos_to_words[p] if
                                                self.word_embeddings.ft_word_similarity(w,
                                                                                        self.theme.split()) > theme_cutoff > self.word_embeddings.gl_word_similarity(
                                                    w, self.theme.split())])
            if False and verbose: print("\n\nfasttext cutting", [w for w in self.pos_to_words[p] if
                                                       self.word_embeddings.ft_word_similarity(w,
                                                                                               self.theme.split()) < theme_cutoff < self.word_embeddings.gl_word_similarity(
                                                           w, self.theme.split())])

            self.pos_to_words[p] = {word: s for (word, s) in self.pos_to_words[p].items() if
                                    self.word_embeddings.both_similarity(word, self.theme.split()) > theme_cutoff}
            if False and verbose: print("ended for", p, len(self.vocab_orig[p]), len(self.pos_to_words[p]),
                              set(self.pos_to_words[p]))
        self.set_meter_pos_dict()

        samples = ["\n".join(random.sample(theme_contexts, theme_lines)) if theme_lines else "" for i in
                   range(4)]  # one for each stanza
        if verbose: print("samples, ", samples)
        # rhymes = []
        # theme = None

        lines = []
        used_templates = []
        choices = []
        # first three stanzas

        self.gpt_past = ""
        line_number = 0
        while line_number < 14:
            if line_number % 4 == 0:
                if verbose: print("\n\nwriting stanza", 1 + line_number / 4)
                # else:
                #    if line_number > 0: print("done")
                #    if len(choices) == 0: print("\nwriting stanza", 1 + line_number/4, end=" ...")
                # alliterated = not alliteration #TODO fix this for later
            lines = lines[:line_number]
            # TODO add rhyme things back in here
            if no_meter:
                meter = {}

            # if r and len()
            # TODO add alliteration back in here

            # self.gpt_past = str(theme_lines and theme.upper() + "\n") + "\n".join(lines) #bit weird but begins with prompt if trying to be themey
            # self.gpt_past = " ".join(theme_words) + "\n" + "\n".join(lines)
            # self.gpt_past = samples[0] + "\n"
            self.gpt_past = samples[0]
            for i in range(len(lines)):
                # if i % 4 == 0: self.gpt_past += samples[i // 4] + "\n"
                # self.gpt_past += lines[i] + "\n"
                if i % 4 == 0: self.gpt_past += samples[i // 4]
                self.gpt_past += lines[i]
            self.reset_letter_words()
            if verbose:
                print("\nwriting line", line_number)

            line = self.write_line_gpt_no_template()  # TODO add parameters back in later

            if line: line_arr = line.split()
            # TODO add some rhyme code back here
            if line:
                line = line.replace(" i ", " I ").replace("\ni", "\nI")
                if verbose: print("wrote line", line)
                if len(lines) % 4 == 0:
                    samp = theme + "\n" + samples[len(lines) // 4] + "\n" + line
                    choices.append((self.gpt.score_line(samp), line))
                else:
                    curr_stanza = "\n".join(lines[len(lines) - (len(lines) % 4):])
                    # line_score = self.gpt.score_line(theme + "\n" + curr_stanza + "\n" + line)
                    line_score = self.gpt.score_line(curr_stanza + "\n" + line)
                    if sum_similarity: line_score *= sum(
                        [self.word_embeddings.ft_word_similarity(w, theme.split()) for w in line.split() if
                         "NN" in self.get_word_pos(w) or "JJ" in self.get_word_pos(w)])
                    choices.append((line_score, line))
                if len(choices) == k:
                    best = min(choices)
                    if verbose:
                        print(choices)
                        print(line_number, ":out of", len(choices), "chose", best)
                    lines.append(best[1])
                    line_number += 1
                    choices = []
                    last = helper.remove_punc(lines[-1].split()[-1])
            else:
                if verbose: print("no line")

        # if not verbose and len(choices) == 0: print("done")
        ret = ("         ---" + theme.upper() + "---       , k=" + str(k) + "\n") if theme else ""
        for cand in range(len(lines)):
            ret += str(lines[cand]) + "\n"
            if ((cand + 1) % 4 == 0): ret += ("\n")
        if verbose: print(ret)

        self.pos_to_words = self.vocab_orig.copy()

        return ret

    def print_beam_history(self, lines=range(14)):
        """
        Prints the beam history of the most recent poem

        Parameters
        ----------
        lines - which of the lines to print for (all by default)

        """

        assert self.line_gen, "no line gen"

        if type(lines) == int:
            lines = [lines]

        for l, hist in enumerate(self.all_beam_histories):
            if l not in lines:
                continue

            print("for line", l, ":")
            for template in hist:
                print("\tfor template", template, ":")
                for num_toks in hist[template]:
                    print("\t\tafter step", num_toks)
                    print("\t", end="\t")
                    for toks, score in hist[template][num_toks]:
                        # print(self.gpt.tokenizer.decode(toks), "(" + str(round(self.gpt.score_tokens(toks), 2)), end="), ")
                        print(self.gpt.tokenizer.decode(toks[:-1]) + "*" + self.gpt.tokenizer.decode(toks[-1]) + "*",
                              "(" + str(round(score, 4)), end="), ")

                    print("")

    def get_top_verb_swaps(self, orig_line, template, rhyme=None, verbose=False):

        line_words = helper.remove_punc(orig_line.replace("'s", "")).split()

        vb_idxs = [i for i in range(len(template)) if "VB" in template[i]]
        vb_pos = [template[i] for i in vb_idxs]

        all_poss_words = set(self.top_common_words + self.get_diff_pos(self.theme, "VB", 500))
        poss_verbs = []
        for i, vb in enumerate(vb_pos):
            poss_verbs.append([w for w in all_poss_words if vb in self.get_word_pos(w) and w not in orig_line])

        if rhyme and "VB" in template[-1]:
            poss_verbs[-1] = [p for p in poss_verbs[-1] if self.rhymes(p, line_words[-1])]


        all_poss_lines = self.get_poss_lines(poss_verbs, orig_line, vb_idxs)

        if verbose:
            print(len(vb_idxs), "verbs gives", len(all_poss_lines), "sentences")

        return [p[0] for p in all_poss_lines]

    def get_poss_lines(self, poss_verbs, orig_line, vb_idxs):

        line_words = helper.remove_punc(orig_line.replace("'s", "")).split()

        all_poss_lines = {(orig_line, tuple(line_words))}
        for i, vb_i in enumerate(vb_idxs):
            new_lines = []
            for old_line, old_line_words in all_poss_lines:
                for new_vb in poss_verbs[i]:
                    new_line = old_line.replace(old_line_words[vb_i], new_vb)
                    new_line_words = helper.remove_punc(new_line.replace("'s", "")).split()
                    new_lines.append((new_line, tuple(new_line_words)))
            all_poss_lines.update(new_lines)

        return all_poss_lines

    def swap_verbs(self, orig_line, template, rhyme=None, seed="", verbose=False):
        if type(template) == str: template = template.split()

        # get every single combination of alternate lines
        #new_lines = self.get_all_verb_swaps(orig_line, template, rhyme, verbose=verbose)
        new_lines = self.get_top_verb_swaps(orig_line, template, rhyme, verbose=verbose)
        assert len(set(new_lines)) == len(new_lines)

        #new_lines = list(set(np.random.choice(new_lines, 10000))) + [orig_line]

        if verbose:
            print("reduced to", len(new_lines))

        # tokenize and score them
        if verbose:
            print("tokenizing", end="...")

        new_line_toks = [self.gpt.tokenizer(n, return_tensors='pt')['input_ids'].to(self.gpt.model.device) for n in
                         new_lines]
        if verbose: print("\tscoring...", end="")

        if seed:
            if verbose: print(" (seed) ", end="")
            old_past = self.gpt.get_past(seed)
        else:
            old_past = None

        if verbose: print(" (all) ")

        new_line_scores = self.gpt.score_all_with_past(new_line_toks, old_past)

        top_idxs = list(np.argsort(new_line_scores))

        #best_score, best_sent = new_line_scores[top_idxs[0]], new_lines[top_idxs[0]]
        best_score, best_sent = self.gpt.score_line(new_lines[top_idxs[0]]), new_lines[top_idxs[0]]

        if verbose:
            print("done: best_score =", best_score, "for '" + best_sent.strip() + "'")
            orig_idx = new_lines.index(orig_line)
            print("compared to", orig_line, new_lines[orig_idx].strip(), new_line_scores[orig_idx])

        return best_sent, best_score



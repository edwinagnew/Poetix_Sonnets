import time


import random
import torch
import json
import numpy as np
import string
import pandas as pd


from py_files import helper

import theme_word_file



#from transformers import BertTokenizer, BertForMaskedLM
#from transformers import RobertaTokenizer, RobertaForMaskedLM

from nltk.corpus import wordnet as wn

import poem_core



class Scenery_Gen(poem_core.Poem):
    def __init__(self, model=None, words_file="saved_objects/tagged_words.p",
                 syllables_file='saved_objects/cmudict-0.7b.txt',
                 extra_stress_file='saved_objects/edwins_extra_stresses.txt',
                 top_file='saved_objects/words/top_words.txt',
                 templates_file="poems/jordan_templates.txt",
                 #templates_file='poems/number_templates.txt',
                 mistakes_file=None):

        #self.templates = [("FROM scJJS scNNS PRP VBZ NN", "0_10_10_1_01_01"),
         #                 ("THAT scJJ scNN PRP VBD MIGHT RB VB", "0_10_10_1_0_10_1"),
          #                ("WHERE ALL THE scNNS OF PRP$ JJ NNS", "0_1_0_10_1_0_10_1"),
           #               ("AND THAT JJ WHICH RB VBZ NN", "0_1_01_0_10_1_01")]

        poem_core.Poem.__init__(self, words_file=words_file, templates_file=templates_file,
                                syllables_file=syllables_file, extra_stress_file=extra_stress_file, top_file=top_file, mistakes_file=mistakes_file)
        if model == "bert":
            self.lang_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.lang_vocab = list(self.tokenizer.vocab.keys())
            self.lang_model.eval()
            self.vocab_to_num = self.tokenizer.vocab

        elif model == "roberta":
            self.lang_model = RobertaForMaskedLM.from_pretrained('roberta-base') # 'roberta-base'
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base') # 'roberta-large'
            with open("saved_objects/roberta/vocab.json") as json_file:
                j = json.load(json_file)
            self.lang_vocab = list(j.keys())
            self.lang_model.eval()
            self.vocab_to_num = {self.lang_vocab[x]: x for x in range(len(self.lang_vocab))}


        else:
            self.lang_model = None

        #with open('poems/kaggle_poem_dataset.csv', newline='') as csvfile:
         #   self.poems = csv.DictReader(csvfile)
        self.poems = list(pd.read_csv('poems/kaggle_poem_dataset.csv')['Content'])
        self.surrounding_words = {}

        #self.gender = random.choice([["he", "him", "his", "himself"], ["she", "her", "hers", "herself"]])

        self.theme_gen = theme_word_file.Theme()

    #override
    def get_pos_words(self,pos, meter=None, rhyme=None, phrase=()):
        """
        Gets all the words of a given POS
        Parameters
        ----------
        pos - the POS you want
        meter - (optional) returns only words which fit the given meter, e.g. 101
        phrase (optional) - returns only words which have a phrase in the dataset. in format ([word1, word2, word3], i) where i is the index of the word to change since the length can be 2 or 3
        """
        #print("oi," , pos, meter, phrase)
        #punctuation management
        punc = [".", ",", ";", "?", ">"]
        #print("here1", pos, meter)
        #if pos[-1] in punc:
        #    p = pos[-1]
        #    if p == ">":
        #        p = random.choice(pos.split("<")[-1].strip(">").split("/"))
        #        pos = pos.split("<")[0] + p
        #    return [word + p for word in self.get_pos_words(pos[:-1], meter=meter, rhyme=rhyme)]
        #print("here", pos, meter, rhyme)
        #similar/repeated word management
        if pos not in self.pos_to_words and "_" in pos:
            sub_pos = pos.split("_")[0]
            word = self.weighted_choice(sub_pos, meter=meter, rhyme=rhyme)
            if not word: input("rhyme broke " + sub_pos + " " + meter + " " + rhyme)
            #word = random.choice(poss)
            if pos.split("_")[1] in string.ascii_lowercase:
                #print("maybe breaking on", pos, word, sub_pos)
                self.pos_to_words[pos] = {word: self.pos_to_words[sub_pos][word]}
            else:
                num = pos.split("_")[1]
                if num not in self.pos_to_words:
                    #self.pos_to_words[pos] = {word:1}
                    self.pos_to_words[num] = word
                else:
                    poss = self.get_pos_words(sub_pos, meter)
                    word = self.pos_to_words[num]
                    self.pos_to_words[pos] = {w: helper.get_spacy_similarity(w, word) for w in poss}
                    return poss

            return [word]
        if rhyme: return [w for w in self.get_pos_words(pos, meter=meter) if self.rhymes(w, rhyme)]
        if len(phrase) == 0 or len(phrase[0]) == 0: return super().get_pos_words(pos, meter=meter)
        else:
            ret = [word for word in self.pos_to_words[pos] if word in self.dict_meters and meter in self.dict_meters[word]]
            phrases = []
            for word in ret:
                phrase[0][phrase[1]] = word
                phrases.append(" ".join(phrase[0]))
            #print(phrases, ret)
            ret = [ret[i] for i in range(len(ret)) if self.phrase_in_poem_fast(phrases[i], include_syns=True)]
            return ret

    #@override
    def last_word_dict(self, rhyme_dict):
        """
        Given the rhyme sets, extract all possible last words from the rhyme set
        dictionaries.

        Parameters
        ----------
        rhyme_dict: dictionary
            Format is   {'A': {tone1 : {similar word: [rhyming words], similar word: [rhyming words], etc.}}, {tone2:{...}}},
                        'B': {tone1 : {similar word: [rhyming words], similar word: [rhyming words], etc.}}, {tone2:{...}}}
                        etc
        Returns
        -------
        dictionary
            Format is {1: ['apple', 'orange'], 2: ['apple', orange] ... }

        """
        scheme = {1: 'A', 2: 'B', 3: 'A', 4: 'B'}
        last_word_dict = {}

        first_rhymes = []
        for i in range(1, len(scheme) + 1):
            if i in [1, 2]:  # lines with a new rhyme -> pick a random key
                last_word_dict[i] = [random.choice(
                    list(rhyme_dict[scheme[i]].keys()))]  # NB ensure it doesnt pick the same as another one
                j = 0
                while not self.suitable_last_word(last_word_dict[i][0], i - 1) or last_word_dict[i][0] in first_rhymes:
                    # or any(rhyme_dict['A'][last_word_dict[i][0]] in rhyme_dict['A'][word] for word in first_rhymes):
                    last_word_dict[i] = [random.choice(list(rhyme_dict[scheme[i]].keys()))]
                    if not any(self.templates[i - 1][1].split("_")[-1] in self.dict_meters[w] for w in
                               rhyme_dict[scheme[i]]):
                        word = last_word_dict[i][0]
                        if self.templates[i - 1][0].split()[-1] in self.get_word_pos(word) and len(
                                self.dict_meters[word][0]) == len(self.templates[i - 1][1].split("_")[-1]) and any(
                                self.suitable_last_word(r, i + 1) for r in rhyme_dict[scheme[i]][word]):
                            self.dict_meters[word].append(self.templates[i - 1][1].split("_")[-1])
                            print("cheated with ", word, " ", self.dict_meters[word],
                                  self.suitable_last_word(word, i - 1))
                    j += 1
                    if j > len(rhyme_dict[scheme[i]]) * 2: input(str(scheme[i]) + " " + str(rhyme_dict[scheme[i]]))
                first_rhymes.append(last_word_dict[i][0])

            if i in [3, 4]:  # lines with an old rhyme -> pick a random value corresponding to key of rhyming couplet
                letter = scheme[i]
                pair = last_word_dict[i - 2][0]
                last_word_dict[i] = [word for word in rhyme_dict[letter][pair] if self.suitable_last_word(word, i - 1)]
                if len(last_word_dict[i]) == 0:
                    print("fuck me", last_word_dict, i, self.templates[i])
                    print(1 / 0)
        return last_word_dict

    #@ovveride
    def suitable_last_word(self, word, line):
        pos = self.templates[line][0].split()[-1].split("sc")[-1]
        meter = self.templates[line][1].split("_")[-1]
        return pos in self.get_word_pos(word) and meter in self.dict_meters[word]

    def write_poem(self, theme="flower", verbose=False, random_templates=True, rhyme_lines=True, checks=()):

        self.gender = random.choice([["i", "me", "my", "mine", "myself"], ["you", "your", "yours", "yourself"],  ["he", "him", "his", "himself"], ["she", "her", "hers", "herself"], ["we", "us", "our", "ours", "ourselves"], ["they", "them", "their", "theirs", "themselves"]])
        self.update_theme_words(self.theme_gen.get_theme_words(theme, verbose=False))

        lines = []
        used_templates = []
        #first three stanzas

        for i in range(3):
            print("\n\nwriting stanza", i)
            next_templates = []
            if not random_templates:
                next_templates = self.templates[i*4:(i+1)*4]
            else:
                while(len(next_templates) < 4):
                    template, meter = self.get_next_template(used_templates)
                    if not rhyme_lines or len(next_templates) < 2 or self.can_rhyme((next_templates[-2][0].split()[-1], next_templates[-2][1].split("_")[-1]), (template.split()[-1], meter.split("_")[-1])):
                        next_templates.append((template, meter))
                        used_templates.append(template)
                        if verbose: print(len(next_templates), template, meter, "worked")
                    else:
                        if verbose: print(next_templates[-2], "wont work with", template, meter)
                        if random.random() < (1/len(self.templates)*2):
                            if verbose: print("so changing", next_templates[-2])
                            #k = i * 4 + len(next_templates) - 2
                            new_t, new_m = self.get_next_template(used_templates[:-2])
                            next_templates[-2] = (new_t, new_m)
                            used_templates[-2] = new_t

            if verbose: print("templates chosen: ", next_templates)
            lines += self.write_stanza(next_templates, rhyme_lines=rhyme_lines, verbose=verbose, checks=checks)

        #fourth stanza
        print("\n\nwriting last couplet")
        if not random_templates:
            (template, meter), (t2, m2) = self.templates[-2:]
        else:
            template, meter = self.get_next_template(used_templates)
            used_templates.append(template)
            t2, m2 = self.get_next_template(used_templates)
            while rhyme_lines and not self.can_rhyme((template.split()[-1], meter.split("_")[-1]), (t2.split()[-1], m2.split("_")[-1])):
                t2, m2 = self.get_next_template(used_templates)
                if verbose: print(t2, m2, "2wont work with", template, meter)
                if random.random() < (1 / len(self.templates) * 2):
                    if verbose: print("2so changing", template, meter)
                    # k = i * 4 + len(next_templates) - 2
                    template, meter = self.get_next_template([])
                    used_templates[0] = template

        lines += self.write_stanza([(template,meter), (t2,m2)], rhyme_lines=rhyme_lines, verbose=verbose, checks=checks)

        print("")
        print("         ---", theme.upper(), "---       ")
        for cand in range(len(lines)):
            print(lines[cand])
            if ((cand + 1) % 4 == 0): print("")

        #return lines


    def write_stanza(self, templates, checks=("RB", "NNS"), rhyme_lines=True, verbose=False):
        """
        Writes a poem from the templates
        Parameters
        ----------
        theme - what the theme words should be about
        checks - a list of POS to check that a phrase exists in the corpus

        Returns
        -------

        """
        lines = []
        rhymes = []

        for line_number, (template, meter) in enumerate(templates):
            if verbose: print("\n", line_number)
            self.reset_letter_words()
            #rhymes = []
            template = template.split()
            meter = meter.split("_")
            #line = self.write_line(line_number, template, meter, rhymes=rhymes)

            r = None
            if rhyme_lines:
                if line_number > 1 or (len(templates) == 2 and line_number == 1):
                    r = rhymes[max(0, line_number-2)]
                    if verbose: print("wr1tting", template, meter, r)
                    line = self.write_line_random(template, meter, rhyme_word=r)
                    if verbose: print("wrot1", line)
                else:
                    line = self.write_line_random(template, meter)
                    if verbose: print("wrote4", line)
                    last_word = line.split()[-1].translate(str.maketrans('', '', string.punctuation))
                    rhyme_temp = templates[min(len(templates)-1, line_number+2)]
                    rhyme_pos = rhyme_temp[0].split()[-1]
                    rhyme_met = rhyme_temp[1].split("_")[-1]
                    while not any(self.rhymes(last_word, w) for w in self.get_pos_words(rhyme_pos,rhyme_met)):
                        if verbose: print("trying", " ".join(template), meter, "again to get a rhyme with", rhyme_pos, rhyme_met)
                        line = self.write_line_random(template, meter)
                        last_word = line.split()[-1].translate(str.maketrans('', '', string.punctuation))
                    if verbose: print("done", line)
                    rhymes.append(last_word)
            else:
                line = self.write_line_random(template, meter)
            for check in checks:
                if check in template:
                    adv = template.index(check)
                    line_arr = line.split()
                    #phrase = []
                    low = max(0,adv-1)
                    if template[low] in self.special_words: low += 1
                    #phrase.append(line_arr[low])
                    #phrase.append(line_arr[adv])
                    high = min(len(line_arr), adv+2)
                    if template[high-1] in self.special_words: high -= 1
                    #phrase.append(line_arr[high])
                    print("checking poem database for", check, " adv ", line, "low: ", low, ", high: ", high, "adv: ", adv, line_arr[low:high])
                    inc_syn = False
                    while not self.phrase_in_poem_fast(line_arr[low:high], include_syns=inc_syn):
                        poss = self.get_pos_words(check, meter=meter[adv], phrase=(line_arr[low:high], line_arr[low:high].index(line_arr[adv])))
                        inc_syn = True
                        if not poss:
                            if not verbose or input("cant find a suitable phrase " + line + str(template) + str(meter) + meter[adv] + " press enter to ignore phrase"):
                                low = high = 0
                            if verbose: print("rewriting line", template, meter, r)
                            line = self.write_line_random(template, meter, rhyme_word=r)
                            inc_syn = False
                            continue
                        line_arr[adv] = random.choice(poss)
                        if verbose: print(check, " updated, line now ", " ".join(line_arr))
                    line = " ".join(line_arr)
            if self.lang_model:
                print("f")
                print("line initially ", line)
                rhyme_set = []
                for r in rhymes:
                    rhyme_set += self.get_rhyme_words(r)
                print("rhyme_set length: ", len(rhyme_set))
                line = self.update_bert(line.strip().split(), meter, template, len(template)/2, rhyme_words=rhyme_set, filter_meter=True, verbose=True)
            if len(lines) % 4 == 0 or lines[-1][-1] in ".?!": line = line.capitalize()
            #print("line now ", line)
            lines.append(line)
            #if template[-1][-1] == ">": template = template[:-1] + [template[-1].split("<")[0] + line[-1]]
            #break
        return lines


    def write_line(self, n, template, meter, rhymes=[], verbose=False):
        #numbers = {}
        line = ""
        punc = ",.;?"
        for i in range(len(template) - 1):
            new_words = []
            scores = []
            pos = template[i]
            #num = -1
            #letter = ""
            new_word = self.weighted_choice(pos, meter=meter[i])
            space = " " * int(new_word not in (punc + "'s"))
            line += space + new_word
        if n == -1 or not rhymes:
            pos = template[-1]#.split("sc")[-1]
            #num = -1
            #letter = ""
            word = self.weighted_choice(pos, meter=meter[-1])

        elif n % 4 < 2:
            word = None
            # high_score = 0
            pos = template[-1]
            #num = -1
            #letter = ""

            rhyme_pos = self.templates[min(13, n + 2)][0].split()[-1]
            rhyme_met = self.templates[min(13, n + 2)][1].split("_")[-1]
            while not word or not any(r in self.get_pos_words(rhyme_pos, meter=rhyme_met) for r in self.get_rhyme_words(word)):
                print("trying to rhyme", pos, meter[-1], ":", word, "with a", rhyme_pos, rhyme_met, self.get_pos_words(rhyme_pos, rhyme_met))
                if word:
                    poss_words = self.get_pos_words(pos, meter[-1])
                    poss_rhymes = {q for w in poss_words for q in self.get_rhyme_words(w)}

                    if not poss_rhymes or not any(r in poss_rhymes for t in self.get_pos_words(rhyme_pos, rhyme_met) for r in self.get_rhyme_words(t)):
                        print("calling in backup")
                        word = random.choice(self.get_backup_words(pos, meter[-1]))
                        continue
                #word = random.choice(self.get_pos_words(pos, meter=meter[-1]))
                word = self.weighted_choice(pos, meter=meter[-1])
                #word = random.choice([w for w in self.get_rhyme_words()])
                """if num in numbers:
                    dist = [helper.get_spacy_similarity(numbers[num], w) for w in poss]
                    word = np.random.choice(poss, p=helper.softmax(dist))
                    if verbose: print("rhyme - weighted choice of ", word, ", related to ", numbers[num], "with prob ", dist[poss.index(word)])
                elif letter in numbers:
                    word = numbers[letter]
                else:
                    word = random.choice(poss) #could get stuck here !!"""
                #if len(poss) == 1: word = poss[0]

                #else: word = np.random.choice(poss, p=helper.softmax([self.pos_to_words[pos][w] for w in poss]))
        else:
            r = (n % 4) - 2
            if n == 13: r = 0
            poss = [rhyme for rhyme in self.get_pos_words(template[-1], meter=meter[-1]) if self.rhymes(rhyme, rhymes[r])]
            if len(poss) == 0:
                poss = [s for s in self.get_backup_words(template[1], meter[-1]) if self.rhymes(s, rhymes[r])]
                if len(poss) == 0:
                    print("couldnt find a rhyme", template[-1], meter[-1])
                    print(1/0)
            word = random.choice(poss)
        line += " " + word
        return line.strip()

    def update_bert(self, line, meter, template, iterations, rhyme_words=[], filter_meter=True, verbose=False, choice = "min"):
        if iterations <= 0: return " ".join(line) #base case
        #TODO deal with tags like ### (which are responsible for actually cool words)
        input_ids = torch.tensor(self.tokenizer.encode(" ".join(line), add_special_tokens=False)).unsqueeze(0) #tokenizes
        tokens = [self.lang_vocab[x] for x in input_ids[0]]
        loss, outputs = self.lang_model(input_ids, masked_lm_labels=input_ids) #masks each token and gives probability for all tokens in each word. Shape num_words * vocab_size
        if verbose: print("loss = ", loss)
        softmax = torch.nn.Softmax(dim=1) #normalizes the probabilites to be between 0 and 1
        outputs = softmax(outputs[0])
        extra_token = ""
        #for word_number in range(0,len(line)-1): #ignore  last word to keep rhyme
        k = tokens.index(self.tokenizer.tokenize(line[-1])[0])  # where the last word begins

        if choice == "rand":
            word_number = out_number = random.choice(np.arange(k))

        elif choice == "min":
            probs = np.array([outputs[i][self.vocab_to_num[tokens[i]]] for i in range(k)])
            word_number = out_number = np.argmin(probs)
            while tokens[out_number].upper() in self.special_words or any(x in self.get_word_pos(tokens[out_number]) for x in ["PRP", "PRP$"]):
                if verbose: print("least likely is unchangable ", tokens, out_number, outputs[out_number][input_ids[0][out_number]])
                probs[out_number] *= 10
                word_number = out_number = np.argmin(probs)

        if len(outputs) > len(line):
            if verbose: print("before: predicting", self.lang_vocab[input_ids[0][out_number]], tokens)
            if tokens[out_number] in line:
                word_number = line.index(tokens[out_number])
                t = 1
                while "##" in self.lang_vocab[input_ids[0][out_number + t]]:
                    extra_token += self.lang_vocab[input_ids[0][out_number + 1]].split("#")[-1]
                    t += 1
                    if out_number + t >= len(input_ids[0]):
                        if verbose: print("last word chosen --> restarting", 1/0)
                        return self.update_bert(line, meter, template, iterations, rhyme_words=rhyme_words, verbose=verbose)
            else:
                sub_tokens = [self.tokenizer.tokenize(w)[0] for w in line]
                while self.lang_vocab[input_ids[0][out_number]] not in sub_tokens: out_number -= 1
                word_number = sub_tokens.index(self.lang_vocab[input_ids[0][out_number]])
                t = 1
                while "##" in self.lang_vocab[input_ids[0][out_number + t]]:
                    extra_token += self.lang_vocab[input_ids[0][word_number + t]].split("#")[-1]
                    t += 1
                    if out_number + t >= len(input_ids[0]):
                        if verbose: print("last word chosen --> restarting", 1/0)
                        return self.update_bert(line, meter, template, iterations, rhyme_words=rhyme_words, verbose=verbose)

            if verbose: print("after: ", out_number, word_number, line, " '", extra_token, "' ")

        #if verbose: print("word number ", word_number, line[word_number], template[word_number], "outnumber:", out_number)

        temp = template[word_number].split("sc")[-1]
        if len(self.get_pos_words(temp)) > 1 and temp not in ['PRP', 'PRP$']: #only change one word each time?
            filt = np.array([int( temp in self.get_word_pos(word) or temp in self.get_word_pos(word + extra_token)) for word in self.lang_vocab])
            if filter_meter and meter: filt *= np.array([int(meter[word_number] in self.get_meter(word) or meter[word_number] in self.get_meter(word + extra_token)) for word in self.lang_vocab])
            predictions = outputs[out_number].detach().numpy() * filt #filters non-words and words which dont fit meter and template

            for p in range(len(predictions)):
                if predictions[p] > 0.001 and self.lang_vocab[p] in rhyme_words:
                    print("weighting internal rhyme '", self.lang_vocab[p], "', orig: ", predictions[p], ", now: ", predictions[p]*5/sum(predictions))
                    predictions[p] *= 5
                """if predictions[p] > 0.001 and self.lang_vocab[p] in theme_words and "sc" in template[word_number]:
                    b = predictions[p]
                    input("change here and for the print")
                    predictions[p] *= theme_words[self.lang_vocab[p]]**2
                    if verbose: print("weighting thematic '", self.lang_vocab[p], "' by ", theme_words[self.lang_vocab[p]], ", now: ", predictions[p]/sum(predictions), ", was: ", b)
"""
            predictions /= sum(predictions)
            if verbose: print("predicting a ", template[word_number], meter[word_number], " for ", word_number, ". min: ", min(predictions), " max: ", max(predictions), "sum: ", sum(predictions), ", ", {self.lang_vocab[p]: predictions[p] for p in range(len(predictions)) if predictions[p] > 0})

            if iterations > 1:
                line[word_number] = np.random.choice(self.lang_vocab, p=predictions)
            else: #greedy for last iteration
                line[word_number] = self.lang_vocab[np.argmax(predictions)]

            print("word now ", line[word_number], "prob: ", predictions[self.lang_vocab.index(line[word_number])])

            if temp not in self.get_word_pos(line[word_number]):
                line[word_number] += extra_token
                if temp not in self.get_word_pos(line[word_number]): #still not valid
                    print("Extra token didnt help ", template[word_number], line[word_number], extra_token)
                    print(1/0)


            if verbose: print("line now", line)
        else:
            if verbose: print("picked ", line[word_number], "which is bad word")
            iterations += 1
        return self.update_bert(line, meter, template, iterations-1, rhyme_words=rhyme_words, verbose=verbose)

    def update_theme_words(self, word_dict, theme=None):
        if theme: word_dict = self.theme_gen.get_theme_words(theme)
        for pos in word_dict:
            self.pos_to_words["sc" + pos] = word_dict[pos]


    def phrase_in_poem(self, words, ret_lines=False, include_syns=False):
        """
        Checks poems database to see if given pair of words exists. If more than two words splits into all pairs
        Parameters
        ----------
        words (string or list) of words to check
        ret_lines (optional) - whether or not to return the lines in which the phrase occurs. Only used for testing
        include_syns (optional) - whether or not to also check for synonyms of phrase words
        """
        if type(words) == list:
            if len(words) == 1: return True
            words = " ".join(words)
        words_arr = words.split()
        if len(words_arr) > 2: return self.phrase_in_poem(words_arr[:2]) and self.phrase_in_poem(words_arr[1:])
        if words_arr[0] == words_arr[1]: return True
        if words_arr[0] in self.gender: return True #?
       # words = " "+ words + " "

        #print("evaluating", words)


        cases = []
        if include_syns:
            syns = []
            for j in range(len(words_arr)):
                syns.append([l.name() for s in wn.synsets(words_arr[j]) for l in s.lemmas() if l.name() in self.dict_meters])
            contenders = [words.split()[0] + " " + w for w in syns[1]]
            contenders += [w + " " + words.split()[1] for w in syns[0]]

        for poem in self.poems:
            poem = " " + poem.lower() + " " #in case its right at beginning or end since we check one before and one after occurence
            i = poem.find(words)
            if i != -1 and poem[i-1] not in string.ascii_lowercase and poem[i+len(words)] not in string.ascii_lowercase: #poem has phrase and characters before and after phrase arent letters (ie spaces or punctuation)
                if not ret_lines: return True
                else:
                    for line in poem.split("\n"):
                        line = " " + line.lower() + " "
                        i = line.find(words)
                        if i != -1 and line[i - 1] not in string.ascii_lowercase and line[i + len(words)] not in string.ascii_lowercase:
                            cases.append(line)
            if include_syns:
                indexes = [poem.find(ws) for ws in contenders]
                if any([indexes[i] != -1 and poem[indexes[i] - 1] not in string.ascii_lowercase and poem[indexes[i] + len(contenders[i])] not in string.ascii_lowercase for i in range(len(indexes))]):
                    if not ret_lines: return True
                    else:
                        correct = contenders[np.argmax(indexes)]
                        for line in poem.split("\n"):
                            line = " " + line.lower() + " "
                            i = line.find(correct)
                            if i != -1 and line[i - 1] not in string.ascii_lowercase and line[i + len(correct)] not in string.ascii_lowercase:
                                cases.append(line)




        if len(cases) == 0: return False
        return cases

    def phrase_in_poem_fast(self, words, include_syns=False):
        if type(words) == list:
            if len(words) <= 1: return True
        else:
            words = words.split()
        if len(words) > 2: return self.phrase_in_poem_fast(words[:2], include_syns=include_syns) and self.phrase_in_poem_fast(words[1:], include_syns=include_syns)
        if words[0] == words[1]: return True
        if words[0][-1] in ",.?;>" or words[1][-1] in ",.?;>": return self.phrase_in_poem_fast((words[0] + " " + words[1]).translate(str.maketrans('', '', string.punctuation)), include_syns=include_syns)
        if words[0] in self.gender: return True  # ?
        # words = " "+ words + " "

        #print("evaluating", words)

        if include_syns:
            syns = []
            for j in words:
                syns.append([l.name() for s in wn.synsets(j) for l in s.lemmas() if l.name() in self.dict_meters])
            contenders = set(words[0] + " " + w for w in syns[1])
            contenders.update([w + " " + words[1] for w in syns[0]])
            #print(words, ": " , contenders)
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
                    if a != -1 and poem[a-1] not in string.ascii_lowercase and poem[a+len(word)] not in string.ascii_lowercase:
                        #print(a, poem[a:a+len(word)])
                        p_words = poem.translate(translator).split() #remove punctuation and split
                        if word in p_words: #not ideal but eg a line which ends with a '-' confuses it
                            a = p_words.index(word)
                            if a - 1 >= 0 and a - 1 < len(p_words): self.surrounding_words[word].add(p_words[a-1])
                            if a + 1 >= 1 and a + 1 < len(p_words): self.surrounding_words[word].add(p_words[a+1])
            return self.phrase_in_poem_fast(words, include_syns=False)

    def get_backup_words(self, pos, meter, words_file="saved_objects/tagged_words.p"):
        if not self.backup_words:
            pc = poem_core.Poem()
            self.backup_words = pc.get_pos_words

        return [p for p in self.backup_words(pos) if meter in self.get_meter(p)]
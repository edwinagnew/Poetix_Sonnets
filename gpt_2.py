import sonnet_basic
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
import random
import nltk

from py_files import helper


class gpt_gen:

    def __init__(self, sonnet_object=None, model="gpt2"):
        if sonnet_object:
            # self.sonnet_words = sonnet_object.get_pos_words
            self.sonnet_object = sonnet_object
        else:
            print("didnt give me a sonnet_method, making one myself")
            self.sonnet_object = sonnet_basic.Sonnet_Gen()
            # self.sonnet_words = self.sonnet_object.get_pos_words

        # if torch.cuda.is_available(): model = "gpt2-large"
        self.model_size = model

        print("loading model")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model)

        self.model = GPT2LMHeadModel.from_pretrained(model)
        if torch.cuda.is_available():
            print("putting to gpu")
            self.model.to('cuda')
        self.mc_model = None
        print("loaded", model)
        self.lemma = nltk.wordnet.WordNetLemmatizer()

        self.line_gen = Line_Generator(self.sonnet_object, self.tokenizer)

    def generation_flex_meter(self, template, meter_dict, seed="", theme_words={}, theme_threshold=0.6, rhyme_word=None,
                              verbose=False, alliteration=None, weight_repetition=True):
        """
        Generates a line of poetry

        Parameters
        ----------
        template
        meter_dict
        seed - (optional) - previous lines in poem
        theme_words - (optional) - dictionary of thematic words in format: {pos: [word1, word2], pos2: [...], ...}
        theme_threshold (optional) - determines whether to only attempt to insert thematic words => if best_theme_token/best_token > theme_threshold
        rhyme_word (optional) - a word which the line has to rhyme with
        verbose
        alliteration (optional) - which possible letters to try and alliterate
        weight_repetition (optional) - boolean. If True, words which occur more than once in the seed or once in the current line (including stems/lemmas) are deweighted to zero

       Procedure:
            0. Preprocessing. If no seed first token is chosen randomly (TODO ?)
            1. Main loop: get next words
                a. Update GPT token scores
                b. Deal with punctuation:
                    i. if a template ends with punctuation(s), it is stored in punc_next for later
                    ii. When the word is finished generating (ie sub_tokens is empty), poss_tokens is set to punc_next
                c. Deal with repetition templates:
                    i. If a template contains a _ and hasnt been seen before, words are chosen normally then filtered for previous repetitions
                    ii. If it has been seen before, that is the option
                d. Potential next words are collected from poem object
            2. Choose next token
                a. If there's only one possible token, correctly choose the spacing and insert it
                b. If theres many options:
                    i. On the first iteration create poss_tokens a list of lists of tokens corresponding to all possible words
                    ii. Create list of all possible tokens for this iteration
                    iii. Filters the gpt scores for only desired tokens and potentially reweights for alliteration, repetition (see weight_repetition) and existing word score
                    iv. normalises the scores and chooses with np.random.choice
                c. post -processing - check if word is complete and then move on

        New procedure:
            0. Preprocessing
            1. Update gpt score
            2. Call line generator for new token
            3. Repeat

        Ideas:

            - scoring words in pos_to_words for theme
            - token alliteration

        """

        """
        0 - preprocessing
        """
        if type(template) != list: template = template.split()

        if verbose: print("tokenizing")
        if not seed or not seed.strip():
            if template[0] in theme_words:
                first_word = random.choice([w for w in theme_words[template[0]] if
                                            any(m in meter_dict for m in self.sonnet_object.get_meter(w))])
            else:
                first_word = random.choice(self.sonnet_object.get_pos_words(template[0], meter=list(meter_dict.keys())))
            first_met = ""
            while first_met not in meter_dict: first_met = random.choice(self.sonnet_object.get_meter(first_word))
            meter_dict = meter_dict[first_met]
            if verbose: print("(seedless) meter dict now", first_met, first_word, meter_dict)
            generated = self.tokenizer.encode(first_word)  # picks first word randomly
            a = 1
        else:
            a = 0
            generated = self.tokenizer.encode(seed)

        context = torch.tensor([generated]).to(self.model.device)
        past = None

        """
        1 - main loop
        """
        self.line_gen.new_line(template, meter_dict, rhyme_word=rhyme_word, theme_words=theme_words,
                               alliteration=alliteration, weight_repetition=weight_repetition,
                               theme_threshold=theme_threshold)
        # i = a
        # while i < len(template):
        for i in range(a, len(template)):
            # 1a - get new tokens
            while True:
                with torch.no_grad():
                    output, past = self.model(context, past_key_values=past, use_cache=True).values()

                if verbose: print(i, "(" + str(len(self.line_gen.sub_tokens)) + ")")

                output += abs(torch.min(output))  # makes all positive

                token = self.line_gen.update(output, i, verbose=verbose)

                generated += [token]  # .tolist()
                # context = token.unsqueeze(0)
                context = torch.tensor(token).unsqueeze(0).to(self.model.device)

                if len(self.line_gen.sub_tokens) == 0 and not self.line_gen.punc_next:
                    break

            # i += int(not punc_next and len(sub_tokens) == 0)

        if verbose: print("tokens: ", generated)
        sequence = self.tokenizer.decode(generated)

        return sequence.replace(seed.strip(), "").strip()

    def score_line(self, line):
        if type(line) == list: return [self.score_line(li.strip()) for li in line]
        input_ids = torch.tensor(self.tokenizer.encode(line, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        with torch.no_grad():
            outputs = self.model(input_ids.to(self.model.device), labels=input_ids.to(self.model.device))
        # loss, logits = outputs[:2]
        return outputs[0].item()

    def get_word_scores(self, line):
        if type(line) == list: return [self.get_word_scores(li.strip()) for li in line]
        input_ids = torch.tensor(self.tokenizer.encode(line, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        with torch.no_grad():
            outputs = self.model(input_ids)
        return torch.tensor([outputs[0][0][max(0, i - 1)][input_ids[0][i]] for i in
                             range(len(input_ids[0]))])  # gets the score of each original word in the line


class Line_Generator:
    def __init__(self, sonnet_object, gpt_tokenizer):
        self.template = None
        self.meter_dict = None
        self.rhyme_word = None

        self.poss = []

        self.sub_tokens = []
        self.poss_tokens = []
        self.theme_tokens = []
        self.theme_words = {}

        self.punc_next = False
        self.repeats = {}

        self.alliteration = None

        self.gpt_tokens = list(gpt_tokenizer.encoder.keys())
        self.gpt_tokenizer = gpt_tokenizer

        self.sonnet_object = sonnet_object

        self.space = ""

        self.weight_repetition = False

        self.theme_threshold = 1

        self.first_lets = set()

        self.lemma = nltk.wordnet.WordNetLemmatizer()

        self.prev_lines = self.curr_line = ""

    def update(self, gpt_output, i, verbose=False):
        """
        Take in gpt_output and return next token
        Parameters
        ----------
        i
        verbose
        gpt_output

        Returns
        -------

        """

        if len(self.sub_tokens) == 0:  # first token of word
            self.get_poss(i, verbose=verbose)

        # all tokens

        # next possible tokens are all the ones which could come after the ones already chosen
        checks = set([p[len(self.sub_tokens)] for p in self.poss_tokens if
                      p[:len(self.sub_tokens)] == self.sub_tokens and len(p) > len(self.sub_tokens)])

        if len(checks) == 1:
            token = checks.pop()
            dist = ws = np.ones(len(self.gpt_tokens))
        else:
            # the same but for theme words
            if self.theme_tokens: theme_checks = set([p[len(self.sub_tokens)] for p in self.theme_tokens if
                                                      p[:len(self.sub_tokens)] == self.sub_tokens and len(p) > len(
                                                          self.sub_tokens)])

            word_scores = self.sonnet_object.pos_to_words[helper.remove_punc(self.template[i].split("_")[0])]

            if any(v != 1 for v in
                   word_scores.values()):  # if words have different scores, get the scores of the relevant words
                score_tokens = {self.gpt_tokenizer.encode(self.space + t)[len(self.sub_tokens)]: v for t, v in
                                word_scores.items() if
                                len(self.gpt_tokenizer.encode(self.space + t)) > len(self.sub_tokens)}
                word_scores = np.array(
                    [score_tokens[t] if t in score_tokens else 0 for t in range(len(self.gpt_tokens))])
            else:
                word_scores = np.ones(len(self.gpt_tokens))

            if len(self.sub_tokens) == 0 and self.alliteration:
                wws = np.array([int(len(x) > 1 and x.strip('Ġ')[0] in self.alliteration) for x in self.gpt_tokens])
                word_scores[wws == 1] *= 2  # maybe make more or += ___
                if verbose: print("alliterating", self.alliteration, sum(wws))

            filt = np.array([int(i in checks) for i in range(len(self.gpt_tokens))]) * word_scores
            ws = gpt_output[..., -1, :].cpu().detach().numpy() * filt
            if ws.shape[0] == 1: ws = ws[0]

            if self.weight_repetition:
                seed_words = helper.remove_punc(self.prev_lines.lower().replace("'s", "")).split()
                lemmas = [self.lemma.lemmatize(word) for word in seed_words]  # gets lemmas for all words in poem

                lemmas_last = [self.lemma.lemmatize(word) for word in helper.remove_punc(self.curr_line.lower().replace("'s", "")).split()]  # gets lemmas for last line in poem

                for j, p in enumerate(self.poss):
                    p_lemma = self.lemma.lemmatize(p)
                    if len(self.sub_tokens) == 0 and (lemmas.count(p_lemma) > 1 or lemmas_last.count(
                            p_lemma) > 0):  # solved - doesnt allow repetition in the same line

                        if self.poss_tokens[j][0] in checks and len(ws.nonzero()) > 1:  # fix
                            repeated_token = self.poss_tokens[j][len(self.sub_tokens)]
                            ws[repeated_token] = 0
                            if verbose: print(p, "was repeated ", lemmas.count(p_lemma) + lemmas_last.count(p_lemma),
                                              "times", "so deweighted", repeated_token,
                                              self.gpt_tokenizer.decode(repeated_token), "\n")
                        else:
                            pass

            theme_scores = []
            if self.theme_tokens: theme_scores = [(ws[x], x) for x in range(len(ws)) if x in theme_checks]

            if verbose: print("max token", np.max(ws), np.argmax(ws), self.gpt_tokens[np.argmax(ws)])
            if verbose: print("top 5", np.argsort(ws)[-5:], [ws[x] for x in np.argsort(ws)[-5:]])
            if verbose and theme_scores: print("best theme", max(theme_scores), self.gpt_tokens[max(theme_scores)[1]])

            if theme_scores and max(theme_scores)[0] / max(ws) > self.theme_threshold:
                if verbose: print("reducing to theme words")
                # ws = [int(x in theme_checks) * ws[x] for x in range(len(words))]
                # print("before", len(ws.nonzero()))
                ws = np.array([int(x in theme_checks) * ws[x] for x in range(len(self.gpt_tokens))])
                if verbose: print("after", len(ws.nonzero()))
            if max(ws) <= 0:
                if verbose: print("something went wrong", max(ws))
                return None

            dist = helper.softmax(ws, exclude_zeros=True)  # , k=np.percentile(words, 0))
            token = np.random.choice(np.arange(len(self.gpt_tokens)), p=dist).item()

        # 2c
        if verbose: print("for ", self.template[i], end=': ')

        if verbose: print("picked " + str(token) + ": '" + str(self.gpt_tokenizer.decode(token)) + "' with prob " + str(
            dist[token]) + " initial score " + str(ws[token]))

        if any(p == self.sub_tokens + [token] for p in self.poss_tokens):

            word = self.gpt_tokenizer.decode(self.sub_tokens + [token]).strip()

            if word not in ",.;?":
                meter = ""
                if verbose: print("getting meter", meter, word, self.sonnet_object.get_meter(word))
                while meter not in self.meter_dict: meter = random.choice(self.sonnet_object.get_meter(word))
                self.meter_dict = self.meter_dict[meter]
                if "_" in self.template[i]:
                    self.repeats[self.template[i]] = word
            self.sub_tokens = []
            self.curr_line += self.space + word
            self.first_lets.add(word[0])
            self.alliteration = self.alliteration if self.alliteration is None or self.alliteration == "s" else self.first_lets
        else:
            self.sub_tokens.append(token)

        # maybe return token here?
        return token

    def get_poss(self, i, verbose=False):
        punc = ",.;?"
        if self.punc_next:
            self.poss = set(self.punc_next)
            self.punc_next = False
        else:
            if self.template[i][-1] == ">":
                self.template[i], self.punc_next = self.template[i].split("<")[0], self.template[i].split("<")[
                    -1].strip(">").split("/")

            elif self.template[i][-1] in punc:
                self.template[i], self.punc_next = self.template[i][:-1], self.template[i][-1]

            # 1c - deal with repetition templates
            if "_" in self.template[i]:
                if self.template[i] in self.repeats:
                    if verbose: print("choosing", self.template[i], "from", self.repeats)
                    self.poss = [self.repeats[self.template[i]]]
                else:
                    self.poss = self.sonnet_object.get_pos_words(self.template[i].split("_")[0],
                                                                 meter=list(self.meter_dict.keys()))
                    self.poss = set([p for p in self.poss if p not in self.repeats.values()])

            # 1d - get potential next words
            else:
                self.poss = set(self.sonnet_object.get_pos_words(self.template[i], meter=list(self.meter_dict.keys())))
            r = None
            if i == len(self.template) - 1 and self.rhyme_word:
                r = self.rhyme_word
                self.poss = set(
                    self.sonnet_object.get_pos_words(self.template[i], meter=list(self.meter_dict.keys()), rhyme=r))

                if verbose: print("restricting to rhymes", self.rhyme_word, self.poss)

        if len(self.poss) == 0:
            if "sc" in self.template[i]:
                if verbose: print("there arent any words so removing sc from", self.template[i])
                self.template[i] = self.template[i].replace("sc", "")

                self.poss = set(
                    self.sonnet_object.get_pos_words(self.template[i], meter=list(self.meter_dict.keys()), rhyme=r))

            if self.sonnet_object.backup_words:  # in case were using byron vocab
                if verbose: print("getting backup words")
                self.poss = set(
                    self.sonnet_object.get_backup_pos_words(self.template[i], list(self.meter_dict.keys()), rhyme=r))

            if len(self.poss) == 0:  # still
                if verbose: print("there arent any words so giving up", self.template[i], self.meter_dict.keys())
                return 1 / 0

        space = self.space = " " * int(list(self.poss)[0] not in punc + "'s" and i > 0)
        if len(self.poss) == 1:
            # 2a - only one option
            # choose token with right spacing
            # if verbose: print(poss, template[i], meter_dict)
            self.poss_tokens = [self.gpt_tokenizer.encode(space + list(self.poss)[0])]
            # token = poss_tokens[0][len(self.sub_tokens)]
            # dist = ws = np.ones(len(self.gpt_tokens))
        else:
            self.poss_tokens = [self.gpt_tokenizer.encode(space + p) for p in self.poss]
            if self.template[i] in self.theme_words:
                self.theme_tokens = [self.gpt_tokenizer.encode(space + p) for p in self.theme_words[self.template[i]] if
                                     p in self.poss]

    def new_line(self, template, meter_dict, rhyme_word=None, theme_words={}, alliteration=None, weight_repetition=True,
                 theme_threshold=0.6):
        self.template = template
        self.meter_dict = meter_dict
        self.rhyme_word = rhyme_word

        self.sub_tokens = []

        # self.punc_next = False

        self.repeats = {}

        self.poss_tokens = []
        self.theme_tokens = []
        self.theme_words = theme_words

        self.alliteration = alliteration
        self.weight_repetition = weight_repetition

        self.theme_threshold = theme_threshold

        self.first_lets = set()

        self.prev_lines += "\n" + self.curr_line
        self.curr_line = ""

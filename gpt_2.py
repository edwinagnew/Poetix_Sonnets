# import sonnet_basic
import poem_core
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
import random
import nltk
import string

from py_files import helper


class gpt_gen:

    def __init__(self, sonnet_object=None, model="gpt2"):
        if sonnet_object:
            # self.sonnet_words = sonnet_object.get_pos_words
            self.sonnet_object = sonnet_object
        else:

            print("didnt give me a sonnet_method, but it's fine, I don't care, I'll just make one myself")
            #self.sonnet_object = sonnet_basic.Sonnet_Gen()
            
            self.sonnet_object = poem_core.Poem()
            # self.sonnet_words = self.sonnet_object.get_pos_words

        # if torch.cuda.is_available(): model = "gpt2-large"
        self.model_size = model

        self.tokenizer = GPT2Tokenizer.from_pretrained(model)

        self.model = GPT2LMHeadModel.from_pretrained(model)
        if torch.cuda.is_available():
            print("putting to gpu")
            self.model.to('cuda')
        self.mc_model = None
        print("loaded", model)
        self.lemma = nltk.wordnet.WordNetLemmatizer()

        self.line_gen = Line_Generator(self.sonnet_object, self.tokenizer)

        self.gpt_tokens = list(self.tokenizer.encoder.keys())
        self.checked_for_rhymes = {}

    def generation_flex_meter(self, template, meter_dict, seed="", theme_words={}, theme_threshold=0.6, rhyme_word=None,
                              verbose=False, alliteration=None, weight_repetition=True, internal_rhymes=[]):
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
            if meter_dict:
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
        print("about to go with", internal_rhymes)
        self.line_gen.new_line(template, meter_dict, rhyme_word=rhyme_word, theme_words=theme_words,
                               alliteration=alliteration, internal_rhymes=internal_rhymes, weight_repetition=weight_repetition,
                               theme_threshold=theme_threshold, prev_lines=seed)
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

    def gen_line_no_template(self, seed="Shall I compare thee to a summer's day?\n", verbose=True):

        meter_dict = self.sonnet_object.get_poss_meters_no_template()

        generated = self.tokenizer.encode(seed)

        context = torch.tensor([generated]).to(self.model.device)
        past = None

        """
        1 - main loop
        """
        self.line_gen.new_line(meter_dict, prev_lines=seed, no_template=True, meter_dict=meter_dict,
                               weight_repetition=False)
        # i = a
        # while i < len(template):
        i = 0  # not needed, but used for keeping track of the line we're on
        while self.line_gen.meter_dict:
            print("starting new iteration of while loop")
            # 1a - get new tokens
            while True:
                print("starting new iteration of second while loop")
                with torch.no_grad():
                    output, past = self.model(context, past_key_values=past, use_cache=True).values()


                if verbose: print(i, "(" + str(len(self.line_gen.sub_tokens)) + ")")

                output += abs(torch.min(output))  # makes all positive
                print(output.max())

                token = self.line_gen.update(output, i, verbose=verbose, no_template=True)

                generated += [token]  # .tolist()
                # context = token.unsqueeze(0)
                context = torch.tensor(token).unsqueeze(0).to(self.model.device)

                if len(self.line_gen.sub_tokens) == 0 and not self.line_gen.punc_next:
                    break
            i += 1
            # i += int(not punc_next and len(sub_tokens) == 0)

        if verbose: print("tokens: ", generated)
        sequence = self.tokenizer.decode(generated)

        return sequence.replace(seed.strip(), "").strip()

    def gen_prose_line(self, seed, verbose=True, internal_rhyme=5, alliteration=True):

        generated = self.tokenizer.encode(seed)

        context = torch.tensor([generated]).to(self.model.device)
        past = None

        found_punc = False

        words = []
        curr_word = []

        a = np.arange(1, 10)
        random.shuffle(a)

        a = a[:internal_rhyme]

        if verbose and internal_rhyme: print("internal rhyming", a)

        i = 0
        while not found_punc and len(words) < 12:
            if verbose: print(i)

            with torch.no_grad():
                output, past = self.model(context, past_key_values=past, use_cache=True).values()

            output += abs(torch.min(output))
            ws = output[..., -1, :].cpu().detach().numpy()

            ws[ws <= 1] = 0


            #if len(curr_word) == 0 and len(words) > 0: #this never happens
            if len(words) > 0 and alliteration and "Ġ" in self.gpt_tokens[np.argmax(ws)]: #best guess at whether new word is starting

                first_lets = set([w[0] for w in words if len(w) > 0])
                for j, token in enumerate(self.gpt_tokens):
                    if len(token) > 1 and token[0] == 'Ġ' and token.strip('Ġ')[0] in first_lets:
                        ws[j] *= 2
                        #print("reweighted token", token, "to", ws[i])

            #internal rhyme very weird
            if i in a and internal_rhyme and "Ġ" in self.gpt_tokens[np.argmax(ws)]:

                wws = np.zeros(len(self.gpt_tokens))
                for w in words:
                    wws += np.array([int(len(token) > 1 and w != token.strip('Ġ') and self.sonnet_object.rhymes(w, token.strip('Ġ'))) for token in self.gpt_tokens])

                ws[wws >= 1] *= 1.5 #doubles the score of all the possible tokens which rhyme with any previous word
                if verbose: print("internal rhyming", internal_rhyme, len(wws.nonzero()[0]))

            scores = helper.softmax(ws, exclude_zeros=True)


            token = np.random.choice(np.arange(len(scores)), p=scores)


            word = self.tokenizer.decode([token])

            if verbose: print(i, token, word)

            if any(p in word.strip() for p in string.punctuation + "\n"):
                found_punc = True

            elif " " in word:
                words.append(self.tokenizer.decode(curr_word).strip())
                curr_word = []

            generated += [token]
            curr_word.append(token)

            #if verbose: print(token)
            context = torch.tensor(token).unsqueeze(0).to(self.model.device)

            i += 1


        sequence = self.tokenizer.decode(generated[-i:])

        return sequence.strip()

    def gen_prose_poem(self, seed=None, n_lines=14, check_threshold=5, verbose=True):
        if seed is None:
            seed = """When day comes we ask ourselves, where can we find light in this never-ending shade?"""  # Gorman poem

        lines = seed + "\n"

        while len(lines.split("\n")) < n_lines:

            l = self.gen_prose_line(seed=lines, verbose=False, internal_rhyme=False)

            if self.score_line(l) < check_threshold:
                lines += l + "\n"
                if verbose: print("got line", len(lines.split("\n")))

        lines = lines.split(seed)[1]

        if verbose: print(lines)

        return lines


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
        self.internal_rhymes = []

        self.gpt_tokens = list(gpt_tokenizer.encoder.keys())
        self.gpt_tokenizer = gpt_tokenizer

        self.sonnet_object = sonnet_object

        self.space = ""

        self.weight_repetition = False

        self.theme_threshold = 1

        self.first_lets = set()

        self.lemma = nltk.wordnet.WordNetLemmatizer()

        self.prev_lines = self.curr_line = ""

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
            if "PRP" in self.template[i] and self.template[i] in self.sonnet_object.pos_to_words:
                self.poss = [p for p in self.sonnet_object.pos_to_words[self.template[i]] if
                             any(m in self.meter_dict for m in self.sonnet_object.get_meter(p))]

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

    def update(self, gpt_output, i, verbose=False, no_template=False):
        """
        Take in gpt_output and return next token
        Parameters
        ----------
        i - word number
        verbose
        gpt_output - token scores

        Returns
        -------

        """

        if len(self.sub_tokens) == 0:  # first token of word
            if no_template:
                self.poss = self.sonnet_object.get_poss_words_no_pos(list(self.meter_dict.keys()))
                space = self.space = " " * int(list(self.poss)[0] and i > 0)
                self.poss_tokens = [self.gpt_tokenizer.encode(space + p) for p in self.poss]
            else:
                self.get_poss(i, verbose=verbose)

        # all tokens

        # next possible tokens are all the ones which could come after the ones already chosen
        len_sub = len(self.sub_tokens)
        checks = set([p[len_sub] for p in self.poss_tokens if
                      p[:len_sub] == self.sub_tokens and len(p) > len_sub]) #creates a list of tokens that could follow the current token based on our vocab

        if len(checks) == 1:
            token = checks.pop()
            dist = ws = np.ones(len(self.gpt_tokens))
        else:
            # the same but for theme words
            if self.theme_tokens: theme_checks = set([p[len_sub] for p in self.theme_tokens if
                                                      p[:len_sub] == self.sub_tokens and len(p) > len_sub])

            if no_template:
                words = self.poss
                word_scores = {}
                for word in words:
                    word_scores[word] = 1
            else:
                word_scores = self.sonnet_object.pos_to_words[helper.remove_punc(self.template[i].split("_")[0])]

            if any(v != 1 for v in
                   word_scores.values()):  # if words have different scores, get the scores of the relevant words
                score_tokens = {self.gpt_tokenizer.encode(self.space + t)[len_sub]: v for t, v in word_scores.items() if
                                len(self.gpt_tokenizer.encode(self.space + t)) > len_sub}
                word_scores = np.array([score_tokens[t] if t in score_tokens else 0 for t in range(len_sub)])
            else:
                if verbose: print("replacing tokens")

                word_scores = np.ones(len(self.gpt_tokens))

            if len_sub == 0 and self.alliteration:
                wws = np.array([int(len(x) > 1 and x.strip('Ġ')[0] in self.alliteration) for x in self.gpt_tokens])
                word_scores[wws == 1] *= 1.5  # maybe make more or += ___
                if verbose: print("alliterating", self.alliteration, sum(wws))

            if self.internal_rhymes:
                #assume internal_rhyme is a set of words that we want to rhyme.
                #for now only increase weight of first token

                rhymes = []
                for w in self.internal_rhymes + self.curr_line.split():
                    if w not in self.sonnet_object.gpt.checked_for_rhymes:
                        temp_rhymes = self.sonnet_object.get_rhyme_words(w)
                        self.sonnet_object.gpt.checked_for_rhymes[w] = temp_rhymes if len(temp_rhymes) > 3 else []
                    rhymes += self.sonnet_object.gpt.checked_for_rhymes[w]

                #print("finding rhymes for", self.internal_rhymes + self.curr_line.split(), len(rhymes), rhymes)

                #tokens = set([self.gpt_tokenizer.tokenize(w)[len_sub] for w in rhymes if w in self.sonnet_object.words_to_pos and len(self.gpt_tokenizer.tokenize((w))) > len_sub])
                all_tokens = [self.gpt_tokenizer.tokenize(self.space + w) for w in rhymes]
                tokens = set([p[len_sub] for p in all_tokens if p[:len_sub] == self.sub_tokens and len(p) > len_sub])
                if verbose: print("hey look at this dumbasses", tokens)

                if verbose: print("number of shared tokens prior to rhyming", len([w for w in tokens if w in self.poss_tokens]))
                #tokens = set([self.gpt_tokenizer.tokenize(w)[len_sub] for w in rhymes if w in self.sonnet_object.words_to_pos])

                wws = np.zeros(len(self.gpt_tokens))

                for t in tokens:
                    wws[self.gpt_tokenizer.encoder[t]] = 1

                orig = word_scores.copy()
                word_scores[wws != 0] *= 2

                if verbose: print("this was the max", orig.argmax(), orig.max())
                if verbose: print("this is the new max", word_scores.argmax(), word_scores.max())

                if verbose: print("internal rhyming", sum(orig != word_scores), len(wws.nonzero()[0]), len(rhymes), self.internal_rhymes)

                if verbose: print("number of shared checks", len([w for w in rhymes if w in self.poss]))


            filt = np.array([int(i in checks) for i in range(len(self.gpt_tokens))]) * word_scores
            print("this is the newer max", filt.argmax(), filt.max())
            #if verbose and self.internal_rhymes: print("number of shared filts", len([w for w in tokens if filt[w] > 0]))

            ws = gpt_output[..., -1, :].cpu().detach().numpy() * filt
            if ws.shape[0] == 1: ws = ws[0]

            if self.weight_repetition:
                ws = self.weight_repeated_words(checks, ws, verbose=verbose)

            print("this is the newerer max", ws.argmax(), ws.max())

            theme_scores = []
            if self.theme_tokens: theme_scores = [(ws[x], x) for x in range(len(ws)) if x in theme_checks]

            if verbose: print("maxest token", np.max(ws), np.argmax(ws), self.gpt_tokens[np.argmax(ws)])
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

            if self.internal_rhymes and self.gpt_tokenizer.decoder[token] in tokens:
                print("i picked a rhymer", token, self.gpt_tokenizer.decode(token))
                forced_words = [sub for sub in rhymes if self.gpt_tokenizer.encode(self.space + sub)[len_sub] == token]
                print("rhymes =", rhymes)
                print("all tokens =", all_tokens)
                viable_words = [f for f in forced_words if f in self.poss]
                if len(viable_words) == 0:
                    pass
                else:
                    self.poss = viable_words
                    if verbose: print("new poss", self.poss)
                    self.poss_tokens = [self.gpt_tokenizer.encode(self.space + p) for p in self.poss]



        # 2c
        if verbose and not no_template: print("for ", self.template[i], end=': ')

        if verbose: print("picked " + str(token) + ": '" + str(self.gpt_tokenizer.decode(token)) + "' with prob " + str(
            dist[token]) + " initial score " + str(ws[token]))

        if any(p == self.sub_tokens + [token] for p in self.poss_tokens):

            word = self.gpt_tokenizer.decode(self.sub_tokens + [token]).strip()

            if word not in ",.;?" and self.meter_dict:
                meter = ""
                # if verbose: print("getting meter", meter, word, self.sonnet_object.get_meter(word))
                while meter not in self.meter_dict: meter = random.choice(self.sonnet_object.get_meter(word))
                self.meter_dict = self.meter_dict[meter]
                if not no_template and "_" in self.template[i]:
                    self.repeats[self.template[i]] = word
            self.sub_tokens = []
            self.curr_line += self.space + word
            self.first_lets.add(word[0])
            self.alliteration = self.alliteration if self.alliteration is None or self.alliteration == "s" else self.first_lets
        else:
            self.sub_tokens.append(token)

        # maybe return token here?
        return token

    def new_line(self, template, meter_dict, rhyme_word=None, theme_words={}, alliteration=None, weight_repetition=True,
                 theme_threshold=0.6, prev_lines="", no_template=False, internal_rhymes=[]):
        if not no_template:
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
        self.weight_repetition = weight_repetition  # False
        self.internal_rhymes = internal_rhymes

        self.theme_threshold = theme_threshold

        self.first_lets = set()

        self.prev_lines = prev_lines
        self.curr_line = ""

    def weight_repeated_words(self, checks, ws, verbose=False):
        seed_words = helper.remove_punc(self.prev_lines.lower().replace("'s", "")).split()
        lemmas = [self.lemma.lemmatize(word) for word in seed_words]  # gets lemmas for all words in poem

        lemmas_last = [self.lemma.lemmatize(word) for word in helper.remove_punc(
            self.curr_line.lower().replace("'s", "")).split()]  # gets lemmas for last line in poem

        for j, p in enumerate(self.poss):
            p_lemma = self.lemma.lemmatize(p)
            if lemmas.count(p_lemma) > 0 or lemmas_last.count(
                    p_lemma) > 0:  # solved - doesnt allow repetition in the same line #changed to 0
                if len(self.sub_tokens) == 0 and self.poss_tokens[j][0] in checks:  # fix
                    if verbose: print(p, "was repeated ", lemmas.count(p_lemma) + lemmas_last.count(p_lemma),
                                      "times")
                    repeated_token = self.poss_tokens[j][0]
                    # ws[repeated_token] = 0
                    dist = 0
                    freq = lemmas.count(p_lemma)
                    if freq > 0:
                        sep = len(lemmas) - lemmas.index(p_lemma)
                        dist = (sep / 150) / freq  # for now

                    ws[repeated_token] *= dist
                    if verbose: print(p, "was repeated ", lemmas.count(p_lemma) + lemmas_last.count(p_lemma),
                                      "times so deweighted", repeated_token, self.gpt_tokenizer.decode(repeated_token),
                                      "\n")
                else:
                    pass

        return ws

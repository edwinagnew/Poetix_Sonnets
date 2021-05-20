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
            # self.sonnet_object = sonnet_basic.Sonnet_Gen()

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
        self.gpt_tokens = list(self.tokenizer.get_vocab().keys())


class Partial_Line:
    def __init__(self, parent, verbose=False):
        # template
        self.parent = parent
        self.template = None
        self.meter_dict = None
        self.poss = []  # all possible words
        self.poss_tokens = []  # tokenization of all possible words

        # poetic_constraints
        # self.alliteration = None
        self.internal_rhymes = [] #previous words to rhyme with
        self.rhyme_finishers = [] #words/tokens that could be used to rhyme

        # syntactic coherance
        self.punc_next = False
        self.repeats = {}

        # history
        self.curr_line = ""
        self.tokens = []  # tokenized version of curr_line
        self.sub_tokens = []
        self.first_lets = set()
        self.past = None  # need to give this to gpt

        # misc
        self.verbose = verbose

    def get_next_word(self):
        word = []
        if not self.template:
            self.get_next_token_no_template()
        else:
            gpt_output = self.get_gpt_scores()
            first_token = self.get_first_token(gpt_output)

            self.update_constraints(first_token, first=True)

            if self.internal_rhymes:
                self.complete_word()

            word.append(first_token)
            while len(self.sub_tokens) != 0:


                gpt_output = self.get_gpt_scores()

                token = self.get_next_token()

                self.update_constraints(token, first=True)

                word.append(token)

        return self.parent.gpt_tokenizer.decode(word)


    def get_gpt_scores(self):
        last_token = self.tokens[-1]
        context = torch.tensor(last_token).unsqueeze(0).to(self.parent.model.device)

        with torch.no_grad():
            output, self.past = self.parent.model(context, past_key_values=self.past, use_cache=True).values()

        output += abs(torch.min(output))

        return output

    def get_next_token_no_template(self):
        print("piss off")

    def get_first_token(self, gpt_output):

        verbose = self.verbose

        i = len(self.curr_line.split())
        self.get_poss(i)

        # next possible tokens are all the ones which could come after the ones already chosen
        checks = set([p[0] for p in self.poss_tokens])  # if p[:len_sub] == self.sub_tokens and len(p) > len_sub])  # creates a list of tokens that could follow the current token based on our vocab

        if len(checks) == 1:
            token = checks.pop()
            dist = ws = np.ones(len(self.parent.gpt_tokens))
        else:
            # the same but for theme words
            if self.theme_tokens: theme_checks = set([p[0] for p in self.theme_tokens])  # if p[:len_sub] == self.sub_tokens and len(p) > len_sub])
            else: theme_checks = []
            # skipping no template
            next_pos = helper.remove_punc(self.template[i].split("_")[0])

            word_scores_dict = self.parent.sonnet_object.pos_to_words[next_pos]

            if any(v != 1 for v in
                   word_scores_dict.values()):  # if words have different scores, get the scores of the relevant words
                # score_tokens = {self.parent.gpt_tokenizer.encode(self.space + t)[0]: v for t, v in word_scores_dict.items()}

                # word_scores = np.array([score_tokens[t][0] if t in score_tokens else 0 for t in range(len(self.parent.gpt_tokens))])
                word_scores = np.ones(len(self.parent.gpt_tokens))
                print("youll probably never get here", 0 / 0)
            else:
                if verbose: print("replacing tokens")

                word_scores = np.ones(len(self.parent.gpt_tokens))

            if self.parent.alliteration:
                word_scores = self.handle_alliteration(word_scores) #DONE

            if self.internal_rhymes:
                self.handle_internal_rhymes(word_scores) #TODO this should change self.rhyme_finishers if successful

            dist = self.create_dist(checks, word_scores, gpt_output, theme_checks)


            token = np.random.choice(np.arange(len(self.parent.gpt_tokens)), p=dist).item()

        return token


    def handle_internal_rhymes(self, word_scores):

        verbose = self.verbose

        # assume internal_rhyme is a set of words that we want to rhyme.
        # for now only increase weight of first token

        rhymes = []
        for w in self.internal_rhymes + self.curr_line.split():
            if w not in self.parent.sonnet_object.gpt.checked_for_rhymes:
                temp_rhymes = self.parent.sonnet_object.get_rhyme_words(w)
                self.parent.sonnet_object.gpt.checked_for_rhymes[w] = temp_rhymes if len(temp_rhymes) > 3 else []

            rhymes += self.parent.sonnet_object.gpt.checked_for_rhymes[w]



        # tokens = set([self.gpt_tokenizer.tokenize(w)[len_sub] for w in rhymes if w in self.sonnet_object.words_to_pos and len(self.gpt_tokenizer.tokenize((w))) > len_sub])
        len_sub = len(self.sub_tokens)

        all_tokens = [self.parent.gpt_tokenizer.tokenize(self.space + w) for w in rhymes]
        tokens = self.rhyme_finishers = set([p[len_sub] for p in all_tokens if p[:len_sub] == self.sub_tokens and len(p) > len_sub])

        if verbose: print("number of shared tokens prior to rhyming", len([w for w in tokens if w in self.poss_tokens]))
        # tokens = set([self.gpt_tokenizer.tokenize(w)[len_sub] for w in rhymes if w in self.sonnet_object.words_to_pos])

        wws = np.zeros(len(self.parent.gpt_tokens))

        for t in tokens:
            wws[self.parent.gpt_tokenizer.encoder[t]] = 1

        orig = word_scores.copy()
        word_scores[wws != 0] *= 2


        if verbose: print("internal rhyming", sum(orig != word_scores), len(wws.nonzero()[0]), len(rhymes),
                          self.internal_rhymes)

        if verbose: print("number of shared checks", len([w for w in rhymes if w in self.poss]))

        return word_scores

    def handle_alliteration(self, word_scores):
        wws = np.array(
            [int(len(x) > 1 and x.strip('Ġ')[0] in self.parent.alliteration) for x in self.parent.gpt_tokens])
        word_scores[wws == 1] *= 1.5  # maybe make more or += ___
        if self.verbose: print("alliterating", self.parent.alliteration, sum(wws))

        return word_scores

    def complete_word(self, first_token, poss_words=None):
        pass

    # also funcitons for each lit device etc.

    # copied from before code
    def get_poss(self, i):
        verbose = self.verbose
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
                    self.poss = self.parent.sonnet_object.get_pos_words(self.template[i].split("_")[0],
                                                                        meter=list(self.meter_dict.keys()))
                    self.poss = set([p for p in self.poss if p not in self.repeats.values()])

            # 1d - get potential next words
            else:
                self.poss = set(
                    self.parent.sonnet_object.get_pos_words(self.template[i], meter=list(self.meter_dict.keys())))

            r = None
            if i == len(self.template) - 1 and self.parent.rhyme_word:
                r = self.parent.rhyme_word
                if self.meter_dict == {}:
                    print("made it here for rhyming w/o meter")
                    self.poss = set(self.parent.sonnet_object.get_pos_words(self.template[i], meter=None, rhyme=r))
                else:
                    self.poss = set(
                        self.parent.sonnet_object.get_pos_words(self.template[i], meter=list(self.meter_dict.keys()),
                                                                rhyme=r))

                if verbose: print("restricting to rhymes", self.parent.rhyme_word, self.poss)

        if len(self.poss) == 0:
            if "PRP" in self.template[i] and self.template[i] in self.parent.sonnet_object.pos_to_words:
                self.poss = [p for p in self.parent.sonnet_object.pos_to_words[self.template[i]] if
                             any(m in self.meter_dict for m in self.parent.sonnet_object.get_meter(p))]

            if "sc" in self.template[i]:
                if verbose: print("there arent any words so removing sc from", self.template[i])
                self.template[i] = self.template[i].replace("sc", "")

                self.poss = set(
                    self.parent.sonnet_object.get_pos_words(self.template[i], meter=list(self.meter_dict.keys()),
                                                            rhyme=r))

            if self.parent.sonnet_object.backup_words:  # in case were using byron vocab
                if verbose: print("getting backup words")
                self.poss = set(
                    self.parent.sonnet_object.get_backup_pos_words(self.template[i], list(self.meter_dict.keys()),
                                                                   rhyme=r))

            if len(self.poss) == 0:  # still
                if verbose: print("there arent any words so giving up", self.template[i], self.meter_dict.keys())
                return 1 / 0

        space = self.space = " " * int(list(self.poss)[0] not in punc + "'s" and i > 0)
        if len(self.poss) == 1:
            # 2a - only one option
            # choose token with right spacing
            # if verbose: print(poss, template[i], meter_dict)
            self.poss_tokens = [self.parent.gpt_tokenizer.encode(space + list(self.poss)[0])]
            # token = poss_tokens[0][len(self.sub_tokens)]
            # dist = ws = np.ones(len(self.gpt_tokens))
        else:
            self.poss_tokens = [self.parent.gpt_tokenizer.encode(space + p) for p in self.poss]
            if self.template[i] in self.parent.theme_words:
                self.theme_tokens = [self.parent.gpt_tokenizer.encode(space + p) for p in
                                     self.parent.theme_words[self.template[i]] if
                                     p in self.poss]

    def weight_repeated_words(self, checks, ws, verbose):

        seed_words = helper.remove_punc(self.parent.prev_lines.lower().replace("'s", "")).split()
        lemmas = [self.parent.lemma.lemmatize(word) for word in seed_words]  # gets lemmas for all words in poem

        lemmas_last = [self.parent.lemma.lemmatize(word) for word in helper.remove_punc(self.curr_line.lower().replace("'s", "")).split()]  # gets lemmas for last line in poem

        for j, p in enumerate(self.poss):
            p_lemma = self.parent.lemma.lemmatize(p)
            if lemmas.count(p_lemma) > 0 or lemmas_last.count(p_lemma) > 0:  # solved - doesnt allow repetition in the same line #changed to 0
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
                                      "times so deweighted", repeated_token, self.parent.gpt_tokenizer.decode(repeated_token),
                                      "\n")
                else:
                    pass

        return ws

    def create_dist(self, checks, word_scores, gpt_output, theme_checks):
        verbose = self.verbose

        filt = np.array([int(i in checks) for i in range(len(self.parent.gpt_tokens))]) * word_scores

        ws = gpt_output[..., -1, :].cpu().detach().numpy() * filt
        if ws.shape[0] == 1: ws = ws[0]

        if self.parent.weight_repetition:
            ws = self.weight_repeated_words(checks, ws, verbose=verbose)

        theme_scores = []

        if self.theme_tokens: theme_scores = [(ws[x], x) for x in range(len(ws)) if x in theme_checks]

        if verbose: print("maxest token", np.max(ws), np.argmax(ws), self.parent.gpt_tokens[np.argmax(ws)])
        if verbose: print("top 5", np.argsort(ws)[-5:], [ws[x] for x in np.argsort(ws)[-5:]])

        if theme_scores and max(theme_scores)[0] / max(ws) > self.parent.theme_threshold:
            if verbose: print("reducing to theme words")

            ws = np.array([int(x in theme_checks) * ws[x] for x in range(len(self.parent.gpt_tokens))])
            if verbose: print("after", len(ws.nonzero()))
        if max(ws) <= 0:
            if verbose: print("something went wrong", max(ws))
            return None

        dist = helper.softmax(ws, exclude_zeros=True)

        return dist

    def update_constraints(self, token, first=False):
        """
        2. Decides whether or not we have just completed a word
        1. Prepare whether or not a word needs to be finished
        """
        i = len(self.curr_line.split())
        if any(p == self.sub_tokens + [token] for p in self.poss_tokens):
            word = self.parent.gpt_tokenizer.decode(self.sub_tokens + [token]).strip()

            if word not in ",.;?" and self.meter_dict:
                meter = ""
                # if verbose: print("getting meter", meter, word, self.sonnet_object.get_meter(word))
                while meter not in self.meter_dict: meter = random.choice(self.parent.sonnet_object.get_meter(word))
                self.meter_dict = self.meter_dict[meter]
                if self.template and "_" in self.template[i]:
                    self.repeats[self.template[i]] = word

            self.sub_tokens = [] #indicates we have moved onto new word
            self.curr_line += self.space + word
            self.first_lets.add(word[0])
            self.parent.alliteration = self.parent.alliteration if (self.parent.alliteration is None or self.parent.alliteration == "s") else self.first_lets
        else:
            self.sub_tokens.append(token)


        if first:
            if self.internal_rhymes and self.parent.gpt_tokenizer.decoder[token] not in self.rhyme_finishers:
                self.rhyme_finishers = []
            #else:
            #    self.any_finishers = True

            if len(self.curr_line.split()) == len(self.template) - 1:
                self.rhyme_finishers = [self.poss_tokens[i] for i in range(len(self.poss_tokens)) if self.poss_tokens[i][0] == token]

        self.tokens.append(token)



class Line_Generator:
    def __init__(self, sonnet_object, gpt_tokenizer):
        # global logistics
        self.rhyme_word = None
        self.space = ""
        self.weight_repetition = False
        self.theme_threshold = 1
        self.lemma = nltk.wordnet.WordNetLemmatizer()
        self.sonnet_object = sonnet_object

        # theme
        self.theme_tokens = []
        self.theme_words = {}

        # gpt
        self.gpt_tokens = list(gpt_tokenizer.get_vocab().keys())
        self.gpt_tokenizer = gpt_tokenizer
        self.prev_lines = ""

        # beam search/multiple k
        self.partial_lines = {}  # dictionary mapping templates to a list of partial line objects

    def new_line(self, template, meter_dict, rhyme_word=None, theme_words={}, alliteration=None, weight_repetition=True,
                 theme_threshold=0.6, prev_lines="", no_template=False, internal_rhymes=[]):

        """if not no_template:
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
        """

        if template not in self.partial_lines:
            self.partial_lines[template] = []

        new_partial = Partial_Line(stuff)

        self.partial_lines[template].append(new_partial)

    def branch(self, n, template):
        """

        Args:
            n: number of branches we wish to create
            template: the template based on which we wish to create partials

        Returns:

        """

    def merge(self, k, template):
        """

        Args:
            k: the number of partials we want returned
            template: the template whose partials we want to merge

        Returns:

        """
        k = min(k, len(self.partial_lines[template]))

    def update_templates(self):
        """
        calls the update_all_partials for each template, partials pair in self.partial lines
        Returns: nothing
        """
        for template in self.partial_lines:
            self.update_all_partials(template)

    def update_all_partials(self, template):
        """
        gets the next word for each partial line in self.partial_lines[template]
        Returns:
        """

    def update(self, gpt_output, i, should_bigram=0, verbose=False, no_template=False):
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
                      p[:len_sub] == self.sub_tokens and len(
                          p) > len_sub])  # creates a list of tokens that could follow the current token based on our vocab

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

            filt = np.array([int(i in checks) for i in range(len(self.gpt_tokens))]) * word_scores
            # if verbose and self.internal_rhymes: print("number of shared filts", len([w for w in tokens if filt[w] > 0]))

            ws = gpt_output[..., -1, :].cpu().detach().numpy() * filt
            if ws.shape[0] == 1: ws = ws[0]

            if self.weight_repetition:
                ws = self.weight_repeated_words(checks, ws, verbose=verbose)

            token = np.random.choice(np.arange(len(self.gpt_tokens)), p=dist).item()

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

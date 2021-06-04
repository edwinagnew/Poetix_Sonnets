# import sonnet_basic
import poem_core
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, GPTNeoForCausalLM
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


        if self.model_size == "custom":
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

            config = GPT2Config.from_json_file('retrained_model/config.json')
            self.model = GPT2LMHeadModel.from_pretrained('retrained_model/pytorch_model.bin',  config=config)
        elif model == "gpt3":

            self.tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
            self.model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model)
            self.model = GPT2LMHeadModel.from_pretrained(model)
        if torch.cuda.is_available():
            print("putting to gpu")
            self.model.to('cuda')
        self.mc_model = None
        print("loaded", model)
        self.gpt_tokens = list(self.tokenizer.get_vocab().keys())

    def score_line(self, line):
        if type(line) == list: return [self.score_line(li.strip()) for li in line]
        input_ids = torch.tensor(self.tokenizer.encode(line, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        with torch.no_grad():
            outputs = self.model(input_ids.to(self.model.device), labels=input_ids.to(self.model.device))
        # loss, logits = outputs[:2]
        return outputs[0].item()


class Partial_Line:
    def __init__(self, parent, template, meter_dict, internal_rhymes=[], verbose=False):
        # template
        self.parent = parent

        self.template = template.split()#.copy()
        self.meter_dict = meter_dict.copy()
        self.poss = []  # all possible words
        self.poss_tokens = []  # tokenization of all possible words

        # poetic_constraints
        # self.alliteration = None
        self.internal_rhymes = internal_rhymes #previous words to rhyme with
        self.rhyme_finishers = [] #words/tokens that could be used to rhyme
        self.theme_tokens = []
        self.alliterated = False

        # syntactic coherance
        self.punc_next = False
        self.repeats = {}

        # history
        self.curr_line = ""
        self.tokens = []  # tokenized version of curr_line
        self.sub_tokens = []
        self.first_lets = set()
        self.past = None  # need to give this to gpt
        self.word_scores = []
        self.template_loc = 0

        # misc
        self.verbose = verbose
        self.line_finished = False


    def write_to_word(self, j):
        while self.template_loc < j:
            self.get_next_word()


    def write_first_word(self):


        self.get_poss(0)

        word = random.choice(list(self.poss))

        word_tokens = self.parent.gpt_tokenizer.encode(word)

        self.sub_tokens = word_tokens[:-1]

        self.update_constraints(word_tokens[-1])

    def get_first_token(self, gpt_output):
        """
        This function gets the first token of a word. It does so factoring in things like internal rhyme, theme, and alliteration. It updates various global parameters to prepare to finish the word.
        Args:
            gpt_output:

        Returns:
            the first token of a word
        """

        verbose = self.verbose

        #i = len(self.curr_line.split())
        i = self.template_loc
        self.get_poss(i)

        # next possible tokens are all the ones which could come after the ones already chosen
        checks = set([p[0] for p in
                      self.poss_tokens])  # if p[:len_sub] == self.sub_tokens and len(p) > len_sub])  # creates a list of tokens that could follow the current token based on our vocab

        if len(checks) == 1:
            token = checks.pop()
            dist = ws = np.ones(len(self.parent.gpt_tokens))
        else:
            # the same but for theme words
            if self.theme_tokens:
                theme_checks = set(
                    [p[0] for p in self.theme_tokens])  # if p[:len_sub] == self.sub_tokens and len(p) > len_sub])
            else:
                theme_checks = []
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
                word_scores = self.handle_alliteration(word_scores)  # DONE

            if self.internal_rhymes:
                self.handle_internal_rhymes(word_scores)  # TODO this should change self.rhyme_finishers if successful

            dist = self.create_dist(checks, word_scores, gpt_output, theme_checks)
            self.word_scores = word_scores

            token = np.random.choice(np.arange(len(self.parent.gpt_tokens)), p=dist).item()

        if self.verbose: print("for", self.template[self.template_loc], "picked " + str(token) + ": '" + str(
            self.parent.gpt_tokenizer.decode(token)) + "' with prob " + str(dist[token]))

        return token


    def get_next_word(self):
        """

        Given a partial line, return a word which fits the next POS from the template (mostly relies on get_first_token and get_next_token)
        -------

        """
        if len(self.tokens) == 0 and len(self.parent.prev_lines) == 0:
            return self.write_first_word()

        assert self.template_loc < len(self.template), self.curr_line

        #word = []
        if not self.template:
            self.get_next_token_no_template() #not implemented yet
        else:
            gpt_output = self.get_gpt_scores()
            first_token = self.get_first_token(gpt_output) #gets the first token

            self.update_constraints(first_token, first=True) #updates all of the global stuff (sub_tokens etc.) accordingly

            if self.rhyme_finishers: #if the word needs to be a rhymer (i.e. last word or first token was the beginning of a rhyme
                self.restrict_words(finishers=self.rhyme_finishers) #updates self.poss_tokens accordingly

            #word.append(first_token)

            while len(self.sub_tokens) != 0: #while the word is not complete

                gpt_output = self.get_gpt_scores()

                token = self.get_next_token(gpt_output) #gets the next token
                self.update_constraints(token, first=False) #updates global stuff accordingly
                #word.append(token)

        #return self.parent.gpt_tokenizer.decode(word) #not really necessary


    def get_gpt_scores(self):
        """
        Uses past and most recent token to get gpt scores

        """



        if self.past:
            last_token = self.tokens[-1]
            context = torch.tensor(last_token).unsqueeze(0).to(self.parent.gpt_model.device)

        else:
            past_text = (self.parent.prev_lines + "\n" + self.curr_line).strip()
            last_tokens = self.parent.gpt_tokenizer.encode(past_text)# + [self.sub_tokens]
            if self.verbose: print(last_tokens)
            context = torch.tensor([last_tokens]).to(self.parent.gpt_model.device)

            if self.verbose: print("context: ", context, context.size())


        with torch.no_grad():
            output, self.past = self.parent.gpt_model(context, past_key_values=self.past, use_cache=True).values()

        output += abs(torch.min(output))

        return output

    def get_next_token(self, gpt_output):
        """
        Call when: you have one token and need to finish a word, but it isn't a special case like internal rhyme or the last word of a line
        Args:
            gpt_output: the output from gpt_2

        Returns:
            a token
        Updates:
            Nothing, it's handled under update contraints
        """


        len_sub = len(self.sub_tokens)
        checks = set([p[len_sub] for p in self.poss_tokens if p[:len_sub] == self.sub_tokens and len(p) > len_sub])  # creates a list of tokens that could follow the current token based on our vocab

        if len(checks) == 1:
            token = checks.pop()
            dist = ws = np.ones(len(self.parent.gpt_tokens))
        else:
            # the same but for theme words
            if self.theme_tokens: theme_checks = set([p[len_sub] for p in self.theme_tokens if p[:len_sub] == self.sub_tokens and len(p) > len_sub])
            else: theme_checks = []
            # skipping no template
            word_scores = self.word_scores
            dist = self.create_dist(checks, word_scores, gpt_output, theme_checks)
            token = np.random.choice(np.arange(len(self.parent.gpt_tokens)), p=dist).item()

        if self.verbose: print("for", self.template[self.template_loc], "picked " + str(token) + ": '" + str(self.parent.gpt_tokenizer.decode(token)) + "' with prob " + str(dist[token]))
        return token

    def get_next_token_no_template(self):
        print("piss off")

    def update_punc(self, punc):
        scores = self.get_gpt_scores() #just to update past

        p = random.choice(punc)
        if self.verbose: print("the punctuation we're trying to add is " + p)

        self.punc_next = False

        return p


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
        valid_tokens = [p for p in all_tokens if p[:len_sub] == self.sub_tokens and len(p) > len_sub]
        tokens = set(p[len_sub] for p in valid_tokens)
        self.rhyme_finishers = all_tokens
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

    def restrict_words(self, finishers):
        """call when you want to complete a word in a specific way"""

        len_sub = len(self.sub_tokens)
        self.poss_tokens = [p for p in self.poss_tokens if p[:len_sub] == self.sub_tokens and p in finishers]

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
                self.poss = set(self.parent.sonnet_object.get_pos_words(self.template[i], meter=list(self.meter_dict.keys())))

            r = None
            if i == len(self.template) - 1 and self.parent.rhyme_word:
                r = self.parent.rhyme_word
                if self.meter_dict == {}:
                    if self.verbose: print("made it here for rhyming w/o meter")
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
                    #if verbose: print(p, "was repeated ", lemmas.count(p_lemma) + lemmas_last.count(p_lemma), "times")
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

        #if verbose: print("maxest token", np.max(ws), np.argmax(ws), self.parent.gpt_tokens[np.argmax(ws)])
        #if verbose: print("top 5", np.argsort(ws)[-5:], [ws[x] for x in np.argsort(ws)[-5:]])

        if theme_scores and max(theme_scores)[0] / max(ws) > self.parent.theme_threshold:
            #if verbose: print("reducing to theme words")

            ws = np.array([int(x in theme_checks) * ws[x] for x in range(len(self.parent.gpt_tokens))])
            if verbose: print("after", len(ws.nonzero()), ws) #TODO this is dodgy
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

        punc = ",.;?"

        did_punc = False

        word = self.parent.gpt_tokenizer.decode(self.sub_tokens + [token]).strip()

        if any(p == self.sub_tokens + [token] for p in self.poss_tokens): #checks if the word is over

            #if self.punc_next: word

            if word not in punc and self.meter_dict:
                meter = "X"
                # if verbose: print("getting meter", meter, word, self.sonnet_object.get_meter(word))
                while meter not in self.meter_dict: meter = random.choice(self.parent.sonnet_object.get_meter(word))
                self.meter_dict = self.meter_dict[meter]
                if self.template and "_" in self.template[self.template_loc]:
                    self.repeats[self.template[self.template_loc]] = word

                self.first_lets.add(word[0])
                self.parent.alliteration = self.parent.alliteration if (
                            self.parent.alliteration is None or self.parent.alliteration == "s") else self.first_lets

            self.word_scores = []
            self.rhyme_finishers = []
            self.sub_tokens = [] #indicates we have moved onto new word

            if self.punc_next:
                did_punc = True


            self.curr_line += self.space + word

            self.template_loc += 1

            if self.template_loc == len(self.template): #line over
                self.line_finished = True
        else:
            self.sub_tokens.append(token)


        if first:
            if self.internal_rhymes and self.parent.gpt_tokenizer.decoder[token] not in self.rhyme_finishers:
                self.rhyme_finishers = []

            if self.template_loc == len(self.template) - 1:
                self.rhyme_finishers = [self.poss_tokens[i] for i in range(len(self.poss_tokens)) if self.poss_tokens[i][0] == token]

        self.tokens.append(token)

        if did_punc:
            punc = self.update_punc(self.punc_next)
            #word += punc
            token = self.parent.gpt_tokenizer.encode(punc)
            self.tokens.append(token)
            self.curr_line += punc






class Line_Generator:
    def __init__(self, sonnet_object, gpt_object, templates=None, meter_dicts=None, rhyme_word=None, theme_words={}, alliteration=None, weight_repetition=True,
                 theme_threshold=0.6, prev_lines="", no_template=False, internal_rhymes=[], k=1, verbose=False):
        # global logistics
        if templates is None:
            templates = []
        if meter_dicts is None:
            meter_dicts = []
        self.rhyme_word = rhyme_word
        self.space = ""
        self.weight_repetition = weight_repetition
        self.theme_threshold = 1
        self.lemma = nltk.wordnet.WordNetLemmatizer()
        self.sonnet_object = sonnet_object
        self.alliteration = alliteration
        self.theme_threshold = theme_threshold
        self.prev_lines = prev_lines
        self.no_template = no_template
        self.internal_rhymes = internal_rhymes
        self.k = k
        self.verbose = verbose

        # theme
        self.theme_tokens = []
        self.theme_words = theme_words

        # gpt
        self.gpt_model = gpt_object.model
        self.gpt_tokens = list(gpt_object.tokenizer.get_vocab().keys())
        self.gpt_tokenizer = gpt_object.tokenizer
        self.prev_lines = ""

        # beam search/multiple k
        self.partial_lines = {}  # dictionary mapping templates to a list of partial line objects
        for i, template in enumerate(templates):
            if meter_dicts[i]:
                self.new_line(template, meter_dicts[i], rhyme_word=self.rhyme_word)


    #def create_partial(self):

    def complete_lines(self):
        for template in self.partial_lines:
            self.update_all_partials(template)
        return self.partial_lines

    def new_line(self, template, meter_dict, rhyme_word=None):
        """ beginning of a new line"""

        self.rhyme_word = rhyme_word

        if template not in self.partial_lines:
            self.partial_lines[template] = []

        for _ in range(self.k):
            new_partial = Partial_Line(self, template, meter_dict, internal_rhymes=self.internal_rhymes, verbose=self.verbose)
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
        return

    def update_all_partials(self, template, j=-1):
        """
        gets the next word for each partial line in self.partial_lines[template]
        Returns:
        """
        for p_l in self.partial_lines[template]:
            if j != -1:
                p_l.write_to_word(j)
            else:
                while not p_l.line_finished:
                    p_l.get_next_word()
        return


    def reset(self):
        self.partial_lines = {}

        self.prev_lines = ""

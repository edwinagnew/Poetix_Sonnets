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
    def __init__(self, parent):
        #template
        self.parent = parent
        self.template = None
        self.meter_dict = None
        self.poss = [] #all possible words
        self.poss_tokens = [] #tokenization of all possible words

        #poetic_constraints
        self.alliteration = None
        self.internal_rhymes = []

        #syntactic coherance
        self.punc_next = False
        self.repeats = {}

        #history
        self.curr_line = ""
        self.sub_tokens = []
        self.first_lets = set()
        self.past = None #need to give this to gpt

    def get_next_token(self):

    def get_next_word(self):

    #also funcitons for each lit device etc.

class Line_Generator:
    def __init__(self, sonnet_object, gpt_tokenizer):
        #global logistics
        self.rhyme_word = None
        self.space = ""
        self.weight_repetition = False
        self.theme_threshold = 1
        self.lemma = nltk.wordnet.WordNetLemmatizer()
        self.sonnet_object = sonnet_object

        #theme
        self.theme_tokens = []
        self.theme_words = {}

        #gpt
        self.gpt_tokens = list(gpt_tokenizer.get_vocab().keys())
        self.gpt_tokenizer = gpt_tokenizer
        self.prev_lines = ""

        #beam search/multiple k
        self.partial_lines = {} #dictionary mapping templates to a list of partial line objects

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
# import sonnet_basic
import poem_core
import torch
import numpy as np
import random
import nltk
import pickle
import time

import helper
from copy import deepcopy

from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config, GPTNeoForCausalLM
from custom_beam_scorer import CustomGPT2Model


class gpt_gen:

    def __init__(self, sonnet_object=None, model="gpt2", fast_tokenizer=True):
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

        if "custom" in self.model_size:
            assert len(
                self.model_size.split()) == 2, "custom should be in the form 'custom fine_tuning/twice_retrained' or something like that"
            model_path = model = self.model_size.split()[1]

            if fast_tokenizer:
                self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            else:
                self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

            config = GPT2Config.from_json_file(model_path + '/config.json')
            self.model = GPT2LMHeadModel.from_pretrained(model_path + '/pytorch_model.bin', config=config)

        elif self.model_size == "gpt3":

            self.tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
            self.model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
        else:
            if fast_tokenizer:
                self.tokenizer = GPT2TokenizerFast.from_pretrained(model)
            else:
                self.tokenizer = GPT2Tokenizer.from_pretrained(model)
            self.model = GPT2LMHeadModel.from_pretrained(model)
        if torch.cuda.is_available():
            print("putting to gpu")
            self.model.to('cuda')
        self.mc_model = None
        print("loaded", model)
        self.gpt_tokens = list(self.tokenizer.get_vocab().keys())
        self.token_sentiment = pickle.load(open("saved_objects/bayes_token.p", "rb"))

        self.checked_for_rhymes = {}

    def score_line(self, line):
        if type(line) == list: return [self.score_line(li.strip()) for li in line]
        input_ids = torch.tensor(self.tokenizer.encode(line, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        # return self.score_tokens(input_ids.to(self.model.device))
        return self.score_tokens_new(input_ids.to(self.model.device))

    def score_tokens(self, tokens):
        if type(tokens) == tuple or type(tokens) == list: tokens = torch.tensor(tokens).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(tokens, labels=tokens)
        # loss, logits = outputs[:2]
        tok_count = 1  # len(input_ids[0]) #TODO - normalise lines
        return outputs[0].item() / tok_count
        # return torch.mean(outputs[1])

    def score_tokens_new(self, input_ids):
        with torch.no_grad():
            s = self.model(input_ids, labels=input_ids)["loss"] #* (input_ids.size(1) - 1)
            return s.item()

    def get_past(self, seed):
        with torch.no_grad():
            seed_toks = self.tokenizer(seed, return_tensors='pt')['input_ids']
            past = self.model(seed_toks, labels=seed_toks, use_cache=True)['past_key_values']
            return past

    def score_all_with_past(self, lines, past):
        scores = []
        for toks in lines:
            with torch.no_grad():
                score = self.model(toks, labels=toks, past_key_values=past)['loss'] #* (toks.size(1) - 1)
                scores.append(score.item())

        return scores

    def get_sentiment(self, word):
        tokenized = self.tokenizer.encode(word)
        sentiment_score = self.token_sentiment[tokenized[0]]
        return sentiment_score

    def gen_no_constraints(self, seed, n_tokens=100):
        input_ids = self.tokenizer.encode(seed, return_tensors='pt')

        output = self.model.generate(input_ids, max_length=n_tokens, no_repeat_ngram_size=2)

        return self.tokenizer.decode(output[0], skip_special_tokens=True)


class Partial_Line:
    def __init__(self, parent, template, meter_dict, internal_rhymes=[], verbose=False):
        # template
        self.parent = parent
        if type(template) == str:
            self.template = template.split()  # .copy()
        else:
            self.template = template.copy()
        if meter_dict is not None:
            self.meter_dict = meter_dict.copy()
        else:
            assert parent.no_meter, "no meter but should meter"
            self.meter_dict = None
        self.poss = []  # all possible words
        self.poss_tokens = []  # tokenization of all possible words

        # poetic_constraints
        # self.alliteration = None
        self.internal_rhymes = internal_rhymes  # previous words to rhyme with
        self.rhyme_finishers = []  # words/tokens that could be used to rhyme
        self.theme_tokens = []
        self.alliterated = False

        # syntactic coherence
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
        self.seed_tokens = []

        # misc
        self.verbose = verbose
        self.line_finished = False
        self.space = ""
        self.score_cache = None

    def write_to_word(self, j):
        while self.template_loc < j:
            self.get_next_word()

    def copy(self):
        # template
        p_copy = Partial_Line(self.parent, self.template, self.meter_dict.copy(), self.internal_rhymes.copy(),
                              self.verbose)

        p_copy.poss = self.poss.copy()  # all possible words
        p_copy.poss_tokens = self.poss_tokens.copy()  # tokenization of all possible words

        # poetic_constraints
        # self.alliteration = None
        p_copy.internal_rhymes = self.internal_rhymes.copy()  # previous words to rhyme with
        p_copy.rhyme_finishers = self.rhyme_finishers.copy()  # words/tokens that could be used to rhyme
        p_copy.theme_tokens = self.theme_tokens.copy()
        p_copy.alliterated = self.alliterated

        # syntactic coherance
        p_copy.punc_next = self.punc_next
        p_copy.repeats = self.repeats.copy()

        # history
        p_copy.curr_line = self.curr_line
        p_copy.tokens = self.tokens.copy()  # tokenized version of curr_line
        p_copy.sub_tokens = self.sub_tokens.copy()
        p_copy.first_lets = self.first_lets.copy()
        p_copy.seed_tokens = self.seed_tokens.copy()

        if self.past:
            p_copy.past = deepcopy(self.past)  # need to give this to gpt
        else:
            p_copy.past = None
        p_copy.word_scores = self.word_scores.copy()
        p_copy.template_loc = self.template_loc

        # misc
        p_copy.line_finished = self.line_finished
        p_copy.space = self.space
        p_copy.punc_next = self.punc_next
        return p_copy

    def write_first_word(self, ret_word=True):

        self.get_poss(0)

        word = random.choice(list(self.poss))  # .capitalize() #TODO - somehow capitalization breaks everything

        word_tokens = self.parent.gpt_tokenizer.encode(word)

        self.sub_tokens = word_tokens[:-1]

        self.update_constraints(word_tokens[-1])

        if ret_word: return word

        return word_tokens

    def get_first_token(self, gpt_output, n_ret=1):
        """
        This function gets the first token of a word. It does so factoring in things like internal rhyme, theme, and alliteration. It updates various global parameters to prepare to finish the word.
        Args:
            gpt_output:

        Returns:
            the first token of a word
        """

        verbose = self.verbose

        # i = len(self.curr_line.split())
        i = self.template_loc
        self.get_poss(i)

        if self.curr_line in self.parent.token_cache:
            assert False, "you temporarily shouldnt be here"
            assert self.parent.branching > 1, "edwin says you shouldnt be here"
            token = self.parent.token_cache[self.curr_line].pop()
            self.word_scores = self.parent.word_scores_cache[self.curr_line]
            if verbose: print("\n got word scores", self.word_scores)
            if verbose: print("got token", token, "from cache on beam: ",
                              len(self.parent.token_cache[self.curr_line]), )
            return token

        # next possible tokens are all the ones which could come after the ones already chosen
        checks = set([p[0] for p in
                      self.poss_tokens])  # if p[:len_sub] == self.sub_tokens and len(p) > len_sub])  # creates a list of tokens that could follow the current token based on our vocab

        if len(checks) == 1:
            tokens = [checks.pop()]
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
                word_scores = np.ones(len(self.parent.gpt_tokens))

            if self.parent.alliteration and not self.parent.phi_score:
                word_scores = self.handle_alliteration(word_scores)  # DONE

            if self.internal_rhymes and not self.parent.phi_score:
                self.handle_internal_rhymes(word_scores)  # TODO this should change self.rhyme_finishers if successful

            dist = self.create_dist(checks, word_scores, gpt_output, theme_checks)
            self.word_scores = word_scores

            # token = np.random.choice(np.arange(len(self.parent.gpt_tokens)), p=dist).item() #old line
            # template_str = self.template if type(self.template) == str else " ".join(self.template)
            # n_branches = len([p for p in self.parent.partial_lines[template_str] if self.curr_line == p.curr_line])

            if self.parent.random_selection:
                token_indices = np.random.choice(np.arange(len(self.parent.gpt_tokens)), n_ret, p=dist)
            else:
                token_indices = [i for i in torch.topk(torch.tensor(dist), n_ret).indices]

            tokens = [i if type(i) == int else i.item() for i in token_indices if dist[i] > 0]
            # if verbose: print("token types: ", type(tokens[0]))

        while len(tokens) < n_ret:
            tokens += [tokens[0]]

        # self.parent.token_cache[self.curr_line] = tokens #word level beam search
        # token = self.parent.token_cache[self.curr_line].pop()
        if n_ret > 1:
            if self.verbose: print("f returning", n_ret, "tokens here", tokens)
            if self.parent.phi_score:
                # assert dist.min() >= 0, "dist is negative still"
                # assert np.isclose(dist.sum(), 1), "not sum to 1"
                # return [(tokens[i], self.score_custom_dists(tokens[i], dist[tokens[i]])) for i in range(len(tokens))]
                return [(tokens[i], dist[tokens[i]]) for i in range(len(tokens))]
            else:
                return [(tokens[i], dist[tokens[i]]) for i in range(len(tokens))]
        token = tokens[0]

        assert type(token) == int, str(type(token)) + " aint no int m9"

        # self.parent.word_scores_cache[self.curr_line] = self.word_scores  #????

        # if verbose and self.parent.token_cache: print("cache size", len(self.parent.token_cache), self.parent.token_cache[self.curr_line])

        if self.verbose: print("for", self.template[self.template_loc], "picked " + str(token) + ": '" + str(
            self.parent.gpt_tokenizer.decode(token)) + "' with prob " + str(dist[token]))

        return token

    def get_next_token(self, gpt_output, n_ret=1):
        """
        Call when: you have one token and need to finish a word, but it isn't a special case like internal rhyme or the last word of a line
        Args:
            gpt_output: the output from gpt_2

        Returns:
            a token
        Updates:
            Nothing, it's handled under update contraints

        Parameters
        ----------
        gpt_output
        n_ret - number of tokens to return
        """

        len_sub = len(self.sub_tokens)
        checks = set([p[len_sub] for p in self.poss_tokens if p[:len_sub] == self.sub_tokens and len(
            p) > len_sub])  # creates a list of tokens that could follow the current token based on our vocab

        if len(checks) == 1:
            tokens = [checks.pop()]
            dist = ws = np.ones(len(self.parent.gpt_tokens))
        else:
            # the same but for theme words
            if self.theme_tokens:
                theme_checks = set(
                    [p[len_sub] for p in self.theme_tokens if p[:len_sub] == self.sub_tokens and len(p) > len_sub])
            else:
                theme_checks = []
            # skipping no template
            word_scores = self.word_scores

            dist = self.create_dist(checks, word_scores, gpt_output, theme_checks)

            assert dist is not None, "dist is None"

            if self.parent.random_selection:
                token_indices = np.random.choice(np.arange(len(self.parent.gpt_tokens)), n_ret, p=dist)
            else:
                token_indices = [i for i in torch.topk(torch.tensor(dist), n_ret).indices]

            tokens = [i if type(i) == int else i.item() for i in token_indices if dist[i] > 0]
            # if self.verbose: print("token types: ", type(tokens[0]))

        while len(tokens) < n_ret:
            tokens += [tokens[0]]

        if n_ret == 1:
            token = tokens[0]
        else:
            if self.verbose: print("returning", n_ret, "tokens here", tokens)
            if self.parent.phi_score:
                # assert dist.min() >= 0, "dist is negative still"
                # assert np.isclose(dist.sum(), 1), "not sum to 1" + str(dist.sum())
                # return [(tokens[i], self.score_custom_dists(tokens[i], dist[tokens[i]])) for i in range(len(tokens))]
                return [(tokens[i], dist[tokens[i]]) for i in range(len(tokens))]
            else:
                return [(tokens[i], dist[tokens[i]]) for i in range(len(tokens))]

        if self.verbose: print("for", self.template[self.template_loc], "picked " + str(token) + ": '" + str(
            self.parent.gpt_tokenizer.decode(token)) + "' with prob " + str(dist[token]))
        return token

    def get_top_tokens(self, num_tokens):
        """
        Gets the best num_tokens and returns a list of tuples (token, score)
        Parameters
        ----------
        num_tokens

        Returns
        -------

        """
        if len(self.tokens) == 0 and len(self.parent.prev_lines.strip()) == 0:
            if self.verbose: print("picking first word first")
            self.write_first_word()

        assert self.template_loc < len(self.template), self.curr_line

        gpt_output = self.get_gpt_scores()

        if len(self.sub_tokens) == 0:  # new word
            tokens = self.get_first_token(gpt_output, n_ret=num_tokens)
        else:
            tokens = self.get_next_token(gpt_output, n_ret=num_tokens)

        assert len(tokens) <= num_tokens, "too many tokens"

        return tokens

    def valid(self, t, len_sub):
        # Gets all the words token could be part of by iterating through poss_tokens and indexing at the correct
        return t[:len_sub] == self.sub_tokens and len(t) > len_sub

    def choose_token(self, token):
        """
        Called from beam search, assigns the next token and updates constraints
        Parameters
        ----------
        token - which token to insert

        Returns
        -------

        """

        is_first = (len(self.sub_tokens) == 0)

        self.update_constraints(token, first=is_first)

        # if self.verbose: print("for", self.template[self.template_loc-1], "picked " + str(token) + ": '" + str(self.parent.gpt_tokenizer.decode(token)) + "' because I was told to", is_first)

    def get_next_word(self):
        """

        Given a partial line, return a word which fits the next POS from the template (mostly relies on get_first_token and get_next_token)
        -------

        """
        if len(self.tokens) == 0 and len(self.parent.prev_lines.strip()) == 0:
            if self.verbose: print("writing first word")
            return self.write_first_word()

        assert self.template_loc < len(self.template), self.curr_line

        # word = []
        if not self.template:
            self.get_next_token_no_template()  # not implemented yet
        else:
            gpt_output = self.get_gpt_scores()
            first_token = self.get_first_token(gpt_output)  # gets the first token

            # if self.verbose: print("got first token", first_token)

            self.update_constraints(first_token,
                                    first=True)  # updates all of the global stuff (sub_tokens etc.) accordingly

            if self.rhyme_finishers:  # if the word needs to be a rhymer (i.e. last word or first token was the beginning of a rhyme
                self.restrict_words(finishers=self.rhyme_finishers)  # updates self.poss_tokens accordingly

            # word.append(first_token)

            while len(self.sub_tokens) != 0:  # while the word is not complete

                gpt_output = self.get_gpt_scores()

                token = self.get_next_token(gpt_output)  # gets the next token
                self.update_constraints(token, first=False)  # updates global stuff accordingly
                # word.append(token)

        # return self.parent.gpt_tokenizer.decode(word) #not really necessary
        if self.parent.branching == 1:
            self.parent.token_cache = {}
            self.parent.word_scores_cache = {}

    def get_gpt_scores(self):
        """
        Uses past and most recent token to get gpt scores

        """

        if self.past:
            last_token = self.tokens[-1]
            context = torch.tensor(last_token).unsqueeze(0).to(self.parent.gpt_model.device)

        else:
            past_text = (self.parent.prev_lines + "\n" + self.curr_line).strip()
            last_tokens = self.parent.gpt_tokenizer.encode(past_text)  # + [self.sub_tokens]
            if self.verbose: print("no past: last_tokens =", last_tokens)
            context = torch.tensor([last_tokens]).to(self.parent.gpt_model.device)

            # if self.verbose: print("context: ", context, context.size())

        with torch.no_grad():
            # output, self.past = self.parent.gpt_model(context, past_key_values=self.past, use_cache=True).values()
            outputs = self.parent.gpt_model(context, past_key_values=self.past, use_cache=True)
            self.past = outputs.past_key_values
            output = outputs.logits

        output += abs(torch.min(output))

        return output

    def get_all_custom(self, gpt_scores, w_alliteration=0, w_repetition=None, w_rhyme=0):

        w_repetition = self.parent.weight_repetition

        # TODO - rewrite individual ones to only consider tokens in pot_tokens and so on (for speedup)

        # TODO - include theme scores

        n = len(self.parent.gpt_tokens)

        if w_alliteration == 0:
            alliteration_vector = torch.zeros(n)
        else:
            alliteration_vector = self.get_alliteration_vector()
        # print("a vector", type(alliteration_vector), alliteration_vector.shape, alliteration_vector.sum(), alliteration_vector)

        if w_repetition == 0:
            repetition_vector = torch.zeros(n)
        else:
            repetition_vector = self.get_repetition_vector()
        # print("re vector", type(repetition_vector), repetition_vector.shape, repetition_vector.sum(), repetition_vector)

        if w_rhyme == 0:
            rhyme_vector = torch.zeros(n)
        else:
            rhyme_vector = self.get_rhyme_vector()
        # print("ry vector", type(rhyme_vector), rhyme_vector.shape, rhyme_vector.sum(), rhyme_vector)

        return gpt_scores + (w_alliteration * alliteration_vector) + (w_repetition * repetition_vector) + (
                w_rhyme * rhyme_vector)

    def get_alliteration_vector(self):

        tokens = self.parent.gpt_tokens

        if not self.parent.alliteration or len(self.sub_tokens) != 0: return np.zeros(len(tokens))

        alliterates = set(["Ġ" + a for a in self.parent.alliteration])

        alliteration_vector = np.array([int(t in alliterates) for t in tokens])

        return alliteration_vector

    def get_repetition_vector(self):
        tokens = self.parent.gpt_tokens

        #if not self.parent.weight_repetition: return np.zeros(len(tokens))

        past_words = helper.remove_punc(self.parent.prev_lines.lower().replace("'s", "")).split()
        lemmas = [self.parent.lemma.lemmatize(word) for word in past_words]  # gets lemmas for all words in poem
        lemmas_last = [self.parent.lemma.lemmatize(word) for word in helper.remove_punc(
            self.curr_line.lower().replace("'s", "")).split()]  # gets lemmas for last line in poem
        # self.score_cache['lemma'][len(self.tokens)] = lemmas, lemmas_last

        r_v = []
        # cacheable?
        for token in tokens:
            pot_words = [list(self.poss)[j] for j, toks in enumerate(self.poss_tokens) if
                         self.valid(toks, len(self.sub_tokens)) and toks[len(self.sub_tokens)] == token]

            scores = []

            for p in pot_words:
                p_lemma = self.parent.lemma.lemmatize(p)
                score = max(0, lemmas.count(p_lemma) - 1) + lemmas_last.count(
                    p_lemma)  # counts occurences of word more than once in poem or at all in current line
                scores.append(score)  # negates since 0 is ideal and the more, the worse

            if len(scores) == 0:
                r_v.append(0)
            else:
                r_v.append(np.mean(scores))

        return np.array(r_v)

    def get_rhyme_vector(self):

        tokens = np.array(self.parent.gpt_tokens)

        if not self.parent.internal_rhymes: return np.zeros(len(tokens))

        rhymes = []
        # gets a list of all words that rhyme with existing words in poem
        for w in self.internal_rhymes + self.curr_line.split():
            if w not in self.parent.sonnet_object.gpt.checked_for_rhymes:
                temp_rhymes = self.parent.sonnet_object.get_rhyme_words(w)
                self.parent.sonnet_object.gpt.checked_for_rhymes[w] = temp_rhymes if len(temp_rhymes) > 3 else []

            rhymes += self.parent.sonnet_object.gpt.checked_for_rhymes[w]

        all_tokens = [self.parent.gpt_tokenizer.tokenize(self.space + w) for w in rhymes]
        # self.rhyme_finishers = all_tokens  # can't remember what this does
        # self.score_cache['rhyme'][self.curr_line] = all_tokens

        # tokens = set([self.gpt_tokenizer.tokenize(w)[len_sub] for w in rhymes if w in self.sonnet_object.words_to_pos and len(self.gpt_tokenizer.tokenize((w))) > len_sub])
        len_sub = len(self.sub_tokens)

        valid_tokens = [p for p in all_tokens if p[:len_sub] == self.sub_tokens and len(p) > len_sub]
        rhyme_tokens = set(p[len_sub] for p in valid_tokens)

        return np.array([int(token in rhyme_tokens) for token in tokens])  # simple as
        # return np.isin(tokens, rhyme_tokens) # slightly slower?

    def adjust_logits(self, logits):
        # template constraints
        assert len(logits.shape) == 1  # 1 dimensional
        assert len(logits) == len(self.parent.gpt_tokens)

        n = logits.shape[0]

        i = self.template_loc
        if i >= len(self.template):
            if self.tokens[-1] == self.parent.newline_token:
                # print("end of the road, forcing", self.parent.eos_token)
                checks = {self.parent.eos_token}
            else:
                # print("adding newline", self.parent.eos_token)
                checks = {self.parent.newline_token}  # if template is over only allow eos token

        else:
            len_sub = len(self.sub_tokens)
            if len_sub == 0:
                self.get_poss(i)

            checks = set([p[len_sub] for p in self.poss_tokens if p[:len_sub] == self.sub_tokens and len(p) > len_sub])

        mask = torch.tensor([1 if i in checks else 0 for i in range(n)])  # could be sped up
        ninfs = torch.ones(n) * -np.inf
        logits = torch.where(mask == torch.zeros(n), ninfs, logits)

        if all(logits == -np.inf):
            print("all bad")
            print("poss tokens", self.poss_tokens)
            print("checks", checks)
            print("subtokens", self.sub_tokens)
            print("mask", mask, len(mask.nonzero()))
            print(1 / 0)

        return self.get_all_custom(logits)

        # remeber to update once

    def get_next_token_no_template(self):
        assert False, ("uh oh, this hasn't been finished yet")

    def update_punc(self, punc):
        if self.past is not None: scores = self.get_gpt_scores()  # just to update past

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
            wws[self.parent.gpt_tokenizer.encode(t)] = 1

        orig = word_scores.copy()
        word_scores[wws != 0] *= 2  # TODO - maybe tweak to avoid pungent beavers

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

        meters = list(self.meter_dict.keys()) if self.meter_dict else None

        next_pos = self.template[i]

        if self.punc_next:
            # if verbose: print("puncing", self.punc_next)
            self.poss = set(self.punc_next)
            self.punc_next = False
        else:
            if next_pos[-1] == ">":
                next_pos, self.punc_next = next_pos.split("<")[0], next_pos.split("<")[
                    -1].strip(">").split("/")

            elif next_pos[-1] in punc:
                next_pos, self.punc_next = self.template[i][:-1], self.template[i][-1]

            # 1c - deal with repetition templates
            if "_" in next_pos:
                if next_pos in self.repeats:
                    if verbose: print("choosing", next_pos, "from", self.repeats)
                    self.poss = [self.repeats[next_pos]]
                else:
                    self.poss = self.parent.sonnet_object.get_pos_words(next_pos.split("_")[0],
                                                                        meter=meters)
                    self.poss = set([p for p in self.poss if p not in self.repeats.values()])

            # 1d - get potential next words
            else:
                self.poss = set(self.parent.sonnet_object.get_pos_words(next_pos, meter=meters))

            r = None
            if i == len(self.template) - 1 and self.parent.rhyme_word:
                r = self.parent.rhyme_word
                meters += [m + "0" for m in meters] # Allowing feminine rhyming as in Hafez
                self.poss = set(self.parent.sonnet_object.get_pos_words(next_pos, meter=meters, rhyme=r))

                #if verbose: print("restricting to rhymes", self.parent.rhyme_word, self.poss)

        if len(self.poss) == 0:
            if "PRP" in self.template[i] and next_pos in self.parent.sonnet_object.pos_to_words:
                if self.meter_dict:
                    self.poss = [p for p in self.parent.sonnet_object.pos_to_words[next_pos] if
                                 any(m in self.meter_dict for m in self.parent.sonnet_object.get_meter(p))]
                else:
                    self.poss = [p for p in self.parent.sonnet_object.pos_to_words[next_pos]]

            if "sc" in next_pos:
                if verbose: print("there arent any words so removing sc from", next_pos)
                next_pos = next_pos.replace("sc", "")

                self.poss = set(
                    self.parent.sonnet_object.get_pos_words(next_pos, meter=meters,
                                                            rhyme=r))

            if self.parent.sonnet_object.backup_words:  # in case were using byron vocab
                if verbose: print("getting backup words")
                self.poss = set(
                    self.parent.sonnet_object.get_backup_pos_words(next_pos, meters,
                                                                   rhyme=r))

            if len(self.poss) == 0:  # still
                if verbose: print("there arent any words so giving up", next_pos, meters)
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
            if next_pos in self.parent.theme_words:
                self.theme_tokens = [self.parent.gpt_tokenizer.encode(space + p) for p in
                                     self.parent.theme_words[next_pos] if
                                     self.parent.theme_words[next_pos] if
                                     p in self.poss]

    def weight_repeated_words(self, checks, ws, verbose):

        past_words = helper.remove_punc(self.parent.prev_lines.lower().replace("'s", "")).split()
        # print("\non repeating... past_words=", past_words)
        lemmas = [self.parent.lemma.lemmatize(word) for word in past_words]  # gets lemmas for all words in poem
        # print("lemmas=", lemmas, "\n")

        lemmas_last = [self.parent.lemma.lemmatize(word) for word in helper.remove_punc(
            self.curr_line.lower().replace("'s", "")).split()]  # gets lemmas for last line in poem

        for j, p in enumerate(self.poss):
            p_lemma = self.parent.lemma.lemmatize(p)
            if lemmas.count(p_lemma) > 0 or lemmas_last.count(
                    p_lemma) > 0:  # solved - doesnt allow repetition in the same line #changed to 0
                if len(self.sub_tokens) == 0 and self.poss_tokens[j][0] in checks:  # fix
                    # if verbose: print(p, "was repeated ", lemmas.count(p_lemma) + lemmas_last.count(p_lemma), "times")
                    repeated_token = self.poss_tokens[j][0]
                    # ws[repeated_token] = 0
                    dist = 0
                    freq = lemmas.count(p_lemma)
                    if freq > 0:
                        sep = len(lemmas) - lemmas.index(p_lemma)
                        dist = (sep / 150) / freq  # for now

                    ws[repeated_token] *= dist
                else:
                    pass

        return ws

    def create_dist(self, checks, word_scores, gpt_output, theme_checks):
        verbose = self.verbose

        filt = np.array(
            [int(i in checks) for i in range(len(self.parent.gpt_tokens))]) * word_scores  # score of valid words

        ws = gpt_output[..., -1, :].cpu().detach().numpy() * filt  # gpt output filtered by validity
        if ws.shape[0] == 1: ws = ws[0]

        if self.parent.weight_repetition and not self.parent.phi_score:
            ws = self.weight_repeated_words(checks, ws, verbose=verbose)

        theme_scores = []

        if not self.parent.phi_score and self.theme_tokens:
            theme_scores = [(ws[x], x) for x in range(len(ws)) if x in theme_checks]
            for i in range(len(theme_scores)):
                if theme_scores[i][1] in self.parent.theme_bayes:
                    theme_scores[i] = (
                        theme_scores[i][0] * self.parent.theme_bayes[theme_scores[i][1]], theme_scores[i][1])

        # if verbose: print("maxest token", np.max(ws), np.argmax(ws), self.parent.gpt_tokens[np.argmax(ws)])
        # if verbose: print("top 5", np.argsort(ws)[-5:], [ws[x] for x in np.argsort(ws)[-5:]])

        if theme_scores and max(theme_scores)[0] / max(ws) > self.parent.theme_threshold:
            # if verbose: print("reducing to theme words")
            ws = np.array([int(x in theme_checks) * ws[x] for x in range(len(self.parent.gpt_tokens))])
            if verbose: print("after", len(ws.nonzero()[0]), ws)

        if max(ws) <= 0:
            if verbose: print("something went wrong", max(ws))
            return 1 / 0

        # dist = helper.softmax(ws, exclude_zeros=True)
        dist = ws

        if self.parent.phi_score:
            # dist = self.get_all_custom(ws)
            a = time.time()
            # dist1 = [self.score_custom_dists(i, ws[i]) for i in range(len(ws))]
            b = time.time()

            # print("one by one took", b-a)

            a = time.time()
            dist = self.get_all_custom(ws)
            b = time.time()

            print("all together took", b - a)
            assert dist is not None, "dist is None"
            assert type(dist) == type(ws), "wrong type"
            assert dist.shape == ws.shape, "wrong length"
            print("dist:", type(dist), dist.shape, dist.sum(), dist)
            # print(np.isclose(dist1, dist).sum(), np.isclose(dist1, dist))
            # input("contine?")

        dist = helper.softmax(dist, exclude_zeros=True)

        #if verbose: print("after2", len(dist.nonzero()[0]), sum(dist), dist)

        assert 0.99 <= sum(dist) <= 1.01, sum(dist)

        return dist

    def update_constraints(self, token, first=False):
        """
        2. Decides whether or not we have just completed a word
        1. Prepare whether or not a word needs to be finished
        """

        punc = ",.;?"

        # did_punc = False

        word = self.parent.gpt_tokenizer.decode(self.sub_tokens + [token]).strip()

        if any(p == self.sub_tokens + [token] for p in self.poss_tokens):  # checks if the word is over

            # if self.punc_next: word

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
            self.sub_tokens = []  # indicates we have moved onto new word

            self.curr_line += self.space + word

            if not self.punc_next: self.template_loc += 1

            if self.template_loc == len(self.template):  # line over
                self.line_finished = True
        else:
            self.sub_tokens.append(token)

        if first:
            if self.internal_rhymes and self.parent.gpt_tokenizer.decode([token]) not in self.rhyme_finishers:
                self.rhyme_finishers = []

            if self.template_loc == len(self.template) - 1:
                self.rhyme_finishers = [self.poss_tokens[i] for i in range(len(self.poss_tokens)) if
                                        self.poss_tokens[i][0] == token]

        self.tokens.append(token)

        if False and did_punc:
            punc = self.update_punc(self.punc_next)
            # word += punc
            token = self.parent.gpt_tokenizer.encode(punc)

            self.tokens += token
            self.curr_line += punc


class Line_Generator:
    def __init__(self, sonnet_object, gpt_object, templates=None, meter_dicts=None, rhyme_word=None, theme_words={},
                 alliteration=None, weight_repetition=True,
                 theme_threshold=0.6, prev_lines="", no_template=False, internal_rhymes=[], verbose=False,
                 theme_tone=None,
                 branching=1, b_inc=1, random_selection=False, beam_score="line", phi_score=False):
        # global logistics
        self.no_meter = False
        if templates is None:
            templates = []
        if meter_dicts is None:
            meter_dicts = []
        elif len(meter_dicts) != len(templates):
            assert len(meter_dicts) == 0, "expected no meters"
            self.no_meter = True
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
        # self.k = k
        self.verbose = verbose
        self.random_selection = random_selection
        self.word_scores_cache = {}

        # gpt
        self.gpt = gpt_object
        self.gpt_model = gpt_object.model
        self.gpt_tokens = list(gpt_object.tokenizer.get_vocab().keys())
        self.gpt_tokenizer = gpt_object.tokenizer
        # self.prev_lines = ""

        # theme
        self.theme_tokens = []
        self.theme_words = theme_words
        if theme_tone == "negative":
            for key in gpt_object.token_sentiment:
                gpt_object.token_sentiment[key] = 1 - gpt_object.token_sentiment[key]
        self.theme_bayes = gpt_object.token_sentiment

        # beam search/multiple k
        self.phi_score = phi_score
        self.partial_lines = {}  # dictionary mapping templates to a list of partial line objects
        self.beam_history = {}
        self.branching = branching
        self.b_inc = b_inc
        self.token_cache = {}
        self.beam_score = beam_score
        for i, template in enumerate(templates):
            if self.no_meter:
                self.new_line(template, None, rhyme_word=self.rhyme_word)
            elif meter_dicts[i]:
                self.new_line(template, meter_dicts[i], rhyme_word=self.rhyme_word)

        # self.

        self.completed_lines = {}

    # def create_partial(self):

    def complete_lines(self):
        if self.branching == 1:
            for template in self.partial_lines:
                self.update_all_partials(template)  # finishes all partial lines if called without a number
        else:
            for template in self.partial_lines:
                k = 0
                while any(not pl.line_finished for pl in self.partial_lines[template]):
                    if self.verbose:
                        print("\nupdating beams")
                        print("word number is currently ", k)
                    self.branch(self.branching - 1, template)
                    if self.verbose: print("finished branching for this iteration")
                    k += self.b_inc
                    k = min(k, len(template.split()))

                    self.update_all_partials(template, j=k)
                    if self.verbose: print("finished updating partials")
                    self.merge(self.branching, template)
                if self.verbose: print("finished the beams for a template")
        return self.partial_lines

    def beam_search_tokenwise(self):
        if self.branching == 1:
            for template in self.partial_lines:
                self.update_all_partials(template)  # finishes all partial lines if called without a number
                self.completed_lines[template] = [h.curr_line for h in self.partial_lines[template] if h.line_finished]
        else:
            for template in self.partial_lines:
                # self.write_all_first_words(template)
                while len(self.partial_lines[template]) > 0:
                    if self.verbose: print("about to update all", len(self.partial_lines[template]), "partials")
                    all_tokens = []
                    for i, hyp in enumerate(self.partial_lines[template].copy()):
                        if not hyp.line_finished:
                            if self.verbose: print("getting top tokens for hyp", i)
                            top_hyp_tokens = hyp.get_top_tokens(self.branching)  # list of tuples of tokens and scores
                            all_tokens.extend([(t[0], t[1], hyp) for t in top_hyp_tokens])
                        else:
                            if template not in self.completed_lines: self.completed_lines[template] = []
                            self.completed_lines[template].append(hyp.curr_line)
                            self.partial_lines[template].remove(hyp)
                            # probably not very beamy but...

                    if self.verbose: print("out of", len(all_tokens), "options", set([a[0] for a in all_tokens]), ":")

                    potential_partials = []
                    for (tok, tok_score, hyp) in all_tokens:
                        assert not hyp.line_finished, "copying a completed line (bad news)"
                        new_hyp = hyp.copy()
                        new_hyp.choose_token(tok)
                        potential_partials.append(new_hyp)

                    num_branches = min(self.branching, len(potential_partials))

                    partial_scores = {}
                    for (i, pot) in enumerate(potential_partials):
                        toks = tuple(pot.tokens)
                        # print(toks, type(toks))
                        if toks not in partial_scores:
                            if self.phi_score or self.beam_score == "token":
                                partial_scores[toks] = all_tokens[i][1]
                                score_direction = "up"
                            elif self.beam_score == "line":
                                partial_scores[toks] = self.gpt.score_tokens(toks)
                                score_direction = "down"
                            elif self.beam_score == "token_gpt":
                                context = torch.tensor(toks[:-1]).to(self.gpt.model.device)
                                if len(context) == 0:
                                    partial_scores[toks] = 0
                                else:
                                    output = self.gpt.model(context)
                                    last_score = output.logits[
                                        -1, toks[-1]]  # gets the gpt score for outputting the final token
                                    partial_scores[toks] = last_score.item()
                                score_direction = "up"
                                # print(partial_scores[toks], type(partial_scores[toks]))
                            else:
                                score_direction = "doesnt matter"
                                print(1 / 0, "was expecting 'line' or 'token' as beam_score")
                            num_toks = len(toks)
                            if num_toks not in self.beam_history[template]:
                                self.beam_history[template][num_toks] = []
                            self.beam_history[template][num_toks].append((toks, partial_scores[toks]))

                    # potential_partials.sort(key=lambda x: self.gpt.score_line(x.curr_line))
                    potential_partials.sort(key=lambda x: partial_scores[tuple(x.tokens)],
                                            reverse=(score_direction == "up"))

                    self.partial_lines[template] = []
                    picked_lines = set()
                    for p_p in potential_partials:
                        if len(self.partial_lines[template]) < num_branches and tuple(
                                p_p.tokens) not in picked_lines:  # the line hasnt already been added
                            self.partial_lines[template].append(p_p)
                            picked_lines.add(tuple(p_p.tokens))

                    if self.verbose: print("reduced back to ", len(self.partial_lines[template]), "hyps",
                                           [self.gpt_tokenizer.decode(p.tokens) for p in self.partial_lines[template]])

        return self.completed_lines

    def new_line(self, template, meter_dict, rhyme_word=None):
        """ beginning of a new line"""

        self.rhyme_word = rhyme_word

        if template not in self.partial_lines:
            self.partial_lines[template] = []
            self.beam_history[template] = {}

        # for _ in range(self.k): #really?
        new_partial = Partial_Line(self, template, meter_dict, internal_rhymes=self.internal_rhymes,
                                   verbose=self.verbose)
        self.partial_lines[template].append(new_partial)

    def branch(self, n, template):
        """

        Args:
            n: number of branches we wish to create
            template: the template based on which we wish to create partials

        Returns:

        """
        poss_lines = self.partial_lines[template]

        new_lines = []
        for pl in poss_lines:
            for i in range(n):
                new_lines.append(pl.copy())
                # print("copying line ", i)
            if self.verbose: print("created", n, "copies")
        self.partial_lines[template].extend(new_lines)
        if self.verbose: print("finished copying for branching")

    def merge(self, k, template):
        """

        Args:
            k: the number of partials we want returned
            template: the template whose partials we want to merge

        Returns:

        """
        for key in self.token_cache:
            assert len(self.token_cache[key]) == 0, str(
                key) + " was not empty. sanity f̴̟̥̰̀̃ą̵̲̘̝̐̋͂̓͗i̶̡̠̻̝͋̾l̶̼͊͆̉̅e̶͖͒̾̾̀͝d̴̮̟̪̪̫̀̄̄͆"

        k = min(k, len(self.partial_lines[template]))
        self.partial_lines[template].sort(key=lambda x: self.gpt.score_line(x.curr_line))
        self.partial_lines[template] = self.partial_lines[template][:k]

        if self.verbose: print("merged down to", len(self.partial_lines[template]), "partials")

        self.token_cache = {}
        self.word_scores_cache = {}

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
        if self.verbose: print("updating", len(self.partial_lines[template]), "partials with template", template,
                               "up to word", j)
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

    def write_all_first_words(self, template):
        if self.verbose: print("writing first words", template)
        for hyp in self.partial_lines[template]:
            word = hyp.write_first_word()
            if self.verbose: print("first word: '" + word + "', loc:", hyp.template_loc, hyp.template[hyp.template_loc])
            num_toks = len(hyp.tokens)
            if num_toks not in self.beam_history[template]:
                self.beam_history[template][num_toks] = []
            self.beam_history[template][num_toks].append((hyp.tokens, 0))


class BeamManager:
    def __init__(self, gpt_size, tokenizer, sonnet_object, verbose=False, weight_repetition=0, rep_penalty=1):

        self.sonnet_object = sonnet_object

        self.partial_lines = []

        self.rhyme_word = None
        self.theme_words = []
        self.alliteration = None
        self.seed = self.prev_lines = ""
        self.internal_rhymes = None
        self.weight_repetition = weight_repetition
        self.rep_penalty = rep_penalty

        self.lemma = nltk.wordnet.WordNetLemmatizer()

        self.verbose = verbose

        if 'custom' in gpt_size:
            # self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

            assert len(
                gpt_size.split()) == 2, "custom should be in the form 'custom fine_tuning/twice_retrained' or something like that"
            model_path = gpt_size.split()[1]

            config = GPT2Config.from_json_file(model_path + '/config.json')
            self.model = CustomGPT2Model.from_pretrained(model_path + '/pytorch_model.bin', config=config)
        else:
            # self.tokenizer = GPT2TokenizerFast.from_pretrained(gpt_size)
            self.model = CustomGPT2Model.from_pretrained(gpt_size)

        self.model_size = gpt_size
        self.model.poem_object = self  # looks dodgy but...

        self.gpt_tokenizer = tokenizer
        if 'eol' in gpt_size:
            special_dict = {'additional_special_tokens': ['[EOL]']}
            self.gpt_tokenizer.add_special_tokens(special_dict)

        self.gpt_tokens = list(self.gpt_tokenizer.get_vocab().keys())

        self.eos_token = self.gpt_tokenizer.eos_token = self.model.config.eos_token_id
        self.newline_token = 198  # new line character

    def reset_to_new_line(self, rhyme_word, theme_words, alliteration, seed, internal_rhymes):

        self.partial_lines = []

        self.rhyme_word = rhyme_word
        self.theme_words = theme_words
        self.alliteration = alliteration
        self.seed = self.prev_lines = seed
        self.internal_rhymes = internal_rhymes

    def generate(self, template, meter, num_beams):

        #if self.verbose: print("generating", num_beams, template)

        new_partial = Partial_Line(self, template, meter, internal_rhymes=self.internal_rhymes, verbose=self.verbose)
        if not self.seed.strip():
            # write first word
            first_word = new_partial.write_first_word()
            self.seed = first_word

            first_tokens = new_partial.tokens

            new_partial.tokens = []
        else:
            first_tokens = []

        input_ids = self.gpt_tokenizer(self.seed, return_tensors="pt").input_ids
        new_partial.seed_tokens = input_ids.tolist()[0]

        for _ in range(num_beams):
            self.partial_lines.append(new_partial.copy())

        # print("input_ids", input_ids)

        outputs = self.model.generate(input_ids=input_ids, num_beams=num_beams, num_return_sequences=num_beams,
                                      early_stopping=False, max_length=input_ids.shape[-1] + 30,
                                      repetition_penalty=self.rep_penalty, no_repeat_ngram_size=4)

        sliced_outputs = [first_tokens + out[len(input_ids[0]):].tolist() for out in outputs]

        # print(sliced_outputs)

        return self.gpt_tokenizer.batch_decode(sliced_outputs, skip_special_tokens=True)

    def get_hyp_from_tokens(self, tokens, exclude=[]):
        tok_list = tokens.tolist()
        for partial in self.partial_lines:
            # all_tokens = partial.seed_tokens + partial.tokens
            all_tokens = partial.tokens
            assert len(all_tokens) == len(tokens), [all_tokens] + [tok_list]
            # print("\n\n", partial.tokens, tokens)
            # print(partial.tokens == tokens)
            if partial not in exclude and all_tokens == tok_list:
                return partial

        print('couldnt get any partials', tok_list, [p.seed_tokens + p.tokens for p in self.partial_lines],
              [e.tokens for e in exclude])
        return 1 / 0

    def update_beams(self, next_tokens, beam_ids, beam_dict):
        # if self.verbose: print("picked", next_tokens, "for", beam_ids, "\n")
        new_beams = []
        for j, id in enumerate(beam_ids):
            old_hyp = beam_dict[id.item()]

            new_hyp = old_hyp.copy()
            new_hyp.choose_token(int(next_tokens[j]))
            new_beams.append(new_hyp)

        self.partial_lines = new_beams

import torch
import math
import random

import heapq

import nltk
from nltk.stem.snowball import SnowballStemmer

from PyDictionary import PyDictionary

import sonnet_basic
import helper

from transformers import BertTokenizer, BertForMaskedLM

"""Implement a*  search based on the generated poem"""
class Sonnet_Improve:
    def __init__(self, syllables_file='saved_objects/cmudict-0.7b.txt',
                 top_file='saved_objects/words/top_words.txt'):

        self.bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.bert_model.eval()

        self.gen_basic = sonnet_basic.Sonnet_Gen()

        with open(top_file) as tf:
            self.top_common_words = [line.strip() for line in tf.readlines()]
        self.dictionary = PyDictionary()

        self.dict_meters = self.gen_basic.dict_meters

        self.goal = 0
        #scores = self.get_bert_score(self.poem)
        #print([s for s in scores])

        #self.tf_idf_dict = pickle.load(open("saved_objects/tf_idf_dict.p", "rb"))
        #self.tf_idf_model = pickle.load(open("saved_objects/tf_idf_model.p", "rb"))


    def gen_poem(self, prompt, print_poem=False):
        self.poem = self.gen_basic.gen_poem_edwin(prompt, print_poem=print_poem)

    def begin_search(self, line, depth, goal=80, verbose=True):
        self.goal = goal
        pq = self.push([], line, verbose=verbose) #maybe using PriorityQueue as it specialises in simultaneous stuff
        print(pq)
        s = self.change_best_line(pq, depth, max_size=10, verbose=verbose)
        print(s)

    def push(self, pq, item, max_size=10, verbose=False):
        if verbose:
            print("pushing ", item, "onto", len(pq), pq)
        if len(pq) > max_size and 1==2: #will fix later
            heapq.heapreplace(pq, (100 - self.evaluate(item, verbose=verbose), item)) #wrong!! removes the best value
        else:
            score = self.evaluate(item, verbose=verbose)
            #line = Line.Line(item, score, self.dict_meters)
            line = item
            heapq.heappush(pq, (100 - score, line))
        return pq

    def change_best_line(self,pq,depth, max_size=10, verbose=False):

        pop = heapq.heappop(pq)
        score = pop[0]
        line = pop[1]#.text
        if depth < 0: return "(depth limit reached) " , pop
        print("change: depth", depth, "line", line, "size", len(pq))

        if score >= self.goal: return line
        words = line.split()
        for i in range(len(words[:-1])):
            new_word = random.choice(self.gen_basic.filtered_nouns_verbs)
            while new_word not in self.dict_meters.keys(): new_word = random.choice(self.gen_basic.filtered_nouns_verbs)
            new_line = words
            new_line[i] = new_word
            new_line = ' '.join(new_line)
            pq = self.push(pq, new_line, max_size=max_size, verbose=verbose)
        return self.change_best_line(pq, depth-1, max_size=max_size, verbose=verbose)

    def evaluate(self, line, verbose=False):
        """
        Heuristics will all, for the time being, be some measure between 0 and 100. 100 is perfect and 0 is terrible.
        The mandatory heuristics are syllables, meter, and the goal will never be surpassed if they are not met
        The others will be coherence and (later) thematicity. The rest are multi-line
        NB Make stricter for first and last lines
        :param line:
        :return:
        """
        iambic = True
        syllables = 0
        if verbose:
            print("eval", line)
        for word in line.split():
            meter = self.dict_meters[word][0]
            syllables += len(meter)
            iambic *= helper.isIambic(word)

        ret = self.get_bert_score(line, verbose=verbose) * 100

        ret -= abs(syllables - 10) * 10 #penalizes 10 per syllable off from 10

        if not iambic:
            return ret * (self.goal/100)
        return ret

    def get_bert_score(self,line, verbose=False, drop_min_max=False, ret="median"):
        """
        returns the coherence of a given line, as measured by using bert to calculate the probability of predicting each indidivual word
        and returning the mean (NB also try min, median, etc)
        Parameters
        ----------
        line (string) - the text you want to evaluate
        verbose (bool) - prints information while calculating score
        drop_min_max (bool) - whether or not to drop the lowest score before calculating the return value. To be investigated

        Returns - the average probability of predicting each word in the text
        -------

        """
        #line+="." #unclear if this is productive

        tensor_input = torch.tensor(self.tokenizer.encode(line)).unsqueeze(0)
        if verbose:
            print("get_bert_scores")
            print(line)
            print(tensor_input)
        output = self.bert_model(tensor_input)
        tensor_input = tensor_input[0]
        predictions = output[0] #gives a probability for each word in the corpus being at each of the words in the line

        softmax = torch.nn.Softmax(dim=1) #normalizes the probabilites to be between 0 and 1
        probs = softmax(predictions[0])
        sentence_score = []
        if verbose:
            print(self.tokenizer.tokenize(line))
            print(tensor_input)
            print("CLS", tensor_input[0] == 101, probs[0][101])
        for i in range(1, len(tensor_input) - 2):#excludes . and [SEP] at end
            if verbose:
                print(tensor_input[i])
                print(probs[i][tensor_input[i]].item())
                if i == 1 and probs[i][tensor_input[i]].item() < 0.5:
                    print("max 1", self.tokenizer.convert_ids_to_tokens(torch.argmax(probs[1]).item()))

            sentence_score.append(probs[i][tensor_input[i]].item())



        if ret == "mean" and drop_min_max: #often, even for good sentences there's one particualry low value which lets the side down. Remove it
            sentence_score.remove(min(sentence_score))
            sentence_score.remove(max(sentence_score))

        if verbose:
            print("sentence scores", sentence_score)

        if ret == "mean":
            return sum(sentence_score)/len(sentence_score) #mean because otherwise initial improvement will be very difficult?
        if ret == "min":
            return min(sentence_score)
        if ret == "median":
            sentence_score = sorted(sentence_score)
            length = len(sentence_score)
            if length % 2 == 0:
                #print(int(length/2 - 1), "+" , int(length/2))
                return (sentence_score[int(length/2) - 1] + sentence_score[int(length/2)])/2
            else:
                #print(math.ceil(length/2))
                return sentence_score[math.ceil(length/2)]

    def generate_metaphor(self, word, avg="median", verbose=False):
        """
        generates a word with metaphorical similarity to the given word by taking key words from dictionary definition, ~averaging~ them
        and returning the closest actual word
        Parameters
        ----------
        word (string) - word to get a metaphor for
        verbose (bool) - print its progress

        Returns
        -------
        a single word with (my interpretation of) metaphorical similarity
        """

        definition = self.dictionary.meaning(word)
        stemmer = SnowballStemmer("english")
        meaning = ""
        for pos in definition:
            for sent in definition[pos]:
                meaning += stemmer.stem(sent)

        if verbose:
            print("generate_metaphor")
            print(definition)

        #summary = word
        summary = ""
        num = 2
        for pos in definition:
            for a in definition[pos][:min(num, len(definition[pos]))]:  # update for more pos's? maybe try with 2 not 3 definitions
                num = max(1,num-1)
                words = a.split(" ")
                i = 0
                while len(words[i]) <= 3 or words[i] in [word] or words[i] in summary: i += 1
                summary += " " + words[i]
                j = 1
                while j < len(words) and len(words[-j]) <= 3 or words[-j] in [word] or words[-j] in summary: j += 1
                if j != len(words): summary += " " + words[-j]

        if verbose:
            print(summary) #summary is a list of key words from top definitions of prompt word

        tokens = self.tokenizer.tokenize(summary)
        input_ids = torch.tensor(self.tokenizer.encode(tokens)).unsqueeze(0)  # Batch size 1
        outputs = self.bert_model(input_ids)
        predictions = outputs[0][0] #predictions gives the probability for predicting each word in the entire vocab for each word in the summary
        #is shape n x 30522 (n=len summary, 30522=vocab size)

        if verbose:
            print(tokens)
            print(predictions.shape)
            print(predictions)

        def is_good_word(stri):
            if verbose: print(stri)
            punctuation = ['-', ':', '+', ',', '～', '`', '!', '.', ',', '?', '*', "\"", "/", "|", "%"]
            if stri in punctuation or '#' in stri:
                return False
            if stri == word or stri in summary:
                return False
            if stemmer.stem(stri) in meaning:
                return False
            if stri in self.top_common_words:
                return False
            pos = nltk.pos_tag([stri])[0][1]
            if 'JJ' in pos :#or 'NN' in pos:  # or 'RB' in pos or 'VB' in pos: #if its a noun, adjective(? or adverb we good just noun for now
                return True
            return False #TODO - see if this word is actually similar to the prompt?

        #creates an average with specified type of length 30522
        if avg == "mean":
            average = torch.mean(predictions[1:-1], 0)
        elif avg == "median":
            average = torch.median(predictions[1:-1], 0).values
        elif avg == "max":
            average = torch.max(predictions[1:-1], 0).values
        elif avg == "lfidf":
            #based off frequnecy of what database?
            print("still working on it ...")
            print(1/0)

        if verbose:
            print("avg", average.shape)

        sorted_average = sorted(average)
        k = 1
        best = torch.argmax(average).item()
        while not is_good_word(self.tokenizer.convert_ids_to_tokens(best)): #simplify by taking into account id's for non words
            k += 1
            y = sorted_average[-k]
            best = ((average == y).nonzero().flatten()).item() #best = average.indexOf(y)

        if verbose:
            print(best)
            print("The metaphor is", self.tokenizer.convert_ids_to_tokens(best))

        return self.tokenizer.convert_ids_to_tokens(best)








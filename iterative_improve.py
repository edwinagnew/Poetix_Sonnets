import torch
import nltk
from PyDictionary import PyDictionary

import sonnet_basic
import helper


from transformers import BertTokenizer,BertForMaskedLM




"""Implement a*  search based on the generated poem"""
class Sonnet_Improve():
    def __init__(self, syllables_file='saved_objects/cmudict-0.7b.txt',
                 top_file='saved_objects/top_words.txt'):

        self.bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.bert_model.eval()

        self.dict_meters = helper.create_syll_dict(syllables_file)

        with open(top_file) as tf:
            self.top_common_words = [line.strip() for line in tf.readlines()]
        self.dictionary = PyDictionary()
        #scores = self.get_bert_score(self.poem)
        #print([s for s in scores])

    def gen_poem(self, prompt, print_poem=False):
        gen_basic = sonnet_basic.Sonnet_Gen()
        self.poem = gen_basic.gen_poem_edwin(prompt, print_poem=print_poem)

        
    def change_line(self,line,depth,goal=80):
        if depth < 0: return line
        if self.evaluate(line, goal) >= goal: return line
        for word in line.split(' '): print(word)


    def evaluate(self, line, goal):
        """
        Heuristics will all, for the time being, be some measure between 0 and 100. 100 is perfect and 0 is terrible.
        The mandatory heuristics are syllables, meter, and the goal will never be surpassed if they are not met
        The other will be coherence. The rest are multi-line
        Make stricter for first and last lines
        :param line:
        :return:
        """
        iambic = True
        syllables = 0
        for word in line.split(' '):
            meter = self.dict_meters[word][0]
            syllables += len(meter)
            iambic *= helper.isIambic(word)

        ret = self.get_bert_score(line)

        if not iambic and syllables != 10:
            return min() / goal
        return min(syllables%14, )

    def get_bert_score(self,line, verbose=True):
        """
        returns the coherence of a given line, as measured by using bert to calculate the probability of predicting each indidivual word
        and returning the mean (NB also try min, median, etc)
        Parameters
        ----------
        line (string) - the text you want to evaluate

        Returns - the average probability of predicting each word in the text
        -------

        """
        tensor_input = torch.tensor(self.tokenizer.encode(line)).unsqueeze(0)

        output = self.bert_model(tensor_input)
        predictions = output[0]

        softmax = torch.nn.Softmax(dim=1)
        probs = softmax(predictions[0])
        sentence_score = []
        if verbose:
            print(self.tokenizer.tokenize(line))
            print(tensor_input[0])
        for i in range(1, len(tensor_input[0]) - 1):
            if verbose:
                print(tensor_input[0][i])
                print(probs[i][tensor_input[0][i]].item())
            sentence_score.append(probs[i][tensor_input[0][i]].item())

        return sum(sentence_score)/len(sentence_score) #mean because otherwise initial improvement will be very difficult
        #return min(sentence_score)

    def generate_metaphor(self, word, verbose=False):
        """
        generates a word with metaphorical similarity to the given word by taking key words from dictionary definition, ~averaging~ them
        and returning the closest actual word
        Parameters
        ----------
        word (string) - word to get a metaphor for
        verbose (bool) - print its progress

        Returns
        -------
        a single word with metaphorical similarity
        """

        definition = self.dictionary.meaning(word)
        if verbose:
            print(definition)

        summary = word
        for a in definition['Noun'][:min(3, len(definition['Noun']))]:  # update for more pos's?
            words = a.split(" ")
            i = 0
            while words[i] in ["a", "the", word, "is"]: i += 1
            summary += " " + words[i]
            while words[-i] in ["a", "the", word, "is"]: i += 1
            summary += " " + words[-i]

        if verbose:
            print(summary)

        tokens = self.tokenizer.tokenize(summary)
        input_ids = torch.tensor(self.tokenizer.encode(tokens)).unsqueeze(0)  # Batch size 1
        outputs = self.bert_model(input_ids)
        predictions = outputs[0][0]

        if verbose:
            print(tokens)
            print(predictions.shape)
            print(predictions)

        def is_good_word(str):
            punctuation = ['-', ':', '+', ',', '～', '`', '!', '.', ',', '?', '*']
            if str in punctuation or '#' in str:
                return False
            if str in self.top_common_words:
                return False
            pos = nltk.pos_tag([str])[0][1]
            if 'NN' in pos:  # or 'JJ' in pos or 'RB' in pos or 'VB' in pos: #if its a noun, adjective(? or adverb we good. just noun for now
                return True
            return False #TODO - see if this word is actually similar to the prompt?

        average = torch.mean(predictions[1:-1, 1:-1], axis=0)  ##improve this!!!
        x = sorted(average)
        k = 1
        best = torch.argmax(average)
        while not is_good_word(self.tokenizer.convert_ids_to_tokens(best.item())):
            k += 1
            y = x[-k]
            best = (average == y).nonzero().flatten() #best = indexOf(y)

        if verbose:
            print(best)
            print("The metaphor is", self.tokenizer.convert_ids_to_tokens(best.item()))

        return self.tokenizer.convert_ids_to_tokens(best.item())





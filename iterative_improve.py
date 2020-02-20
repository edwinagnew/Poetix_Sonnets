import math
import torch

import sonnet_basic
import helper

from pytorch_pretrained_bert import BertTokenizer,BertForMaskedLM


"""Implement a*  search based on the generated poem"""
class Sonnet_Improve():
    def __init__(self, prompt, syllables_file='saved_objects/cmudict-0.7b.txt'):

        self.BertMaskedLM = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.dict_meters = helper.create_syll_dict(syllables_file)


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

    def get_bert_score(self,sentence):
        ret = []
        for s in sentence:
            print(s)
            tokenize_input = self.BertTokenizer.tokenize(sentence)
            tensor_input = torch.tensor([self.BertTokenizer.convert_tokens_to_ids(tokenize_input)])
            predictions = self.BertMaskedLM(tensor_input)
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(predictions.squeeze(), tensor_input.squeeze()).data
            ret.append(s, (8 - loss) * 100/8) #analyse! look for consistency
        return ret


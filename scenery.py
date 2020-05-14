import pickle
import random
import torch
import numpy as np

from py_files import helper

from transformers import BertTokenizer, BertForMaskedLM


class Scenery_Gen():
    def __init__(self, postag_file='saved_objects/postag_dict_all+VBN.p',
                 syllables_file='saved_objects/cmudict-0.7b.txt',
                 extra_stress_file='saved_objects/edwins_extra_stresses.txt'):
        self.templates = [("FROM JJS NNS PRP VBP NN" , "0_10_10_1_01_01"),
                          ("THAT RB NN POS VBD MIGHT RB VB", "0_10_10__1_0_10_1"),
                          ("WHERE ALL THE NN OF PRP$ JJ NNS", "0_1_0_10_1_0_10_1"),
                          ("AND THAT JJ WHICH RB VBZ VB", "0_1_01_0_10_1_01")]
        with open(postag_file, 'rb') as f:
            postag_dict = pickle.load(f)
        self.pos_to_words = postag_dict[1]
        self.words_to_pos = postag_dict[2]

        self.special_words = helper.get_finer_pos_words()

        self.dict_meters = helper.create_syll_dict([syllables_file], extra_stress_file)

        self.bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.bert_model.eval()

        self.bert_vocab = list(self.tokenizer.vocab.keys())

    def get_word_pos(self, word):
        """
        Get the set of POS category of a word. If we are unable to get the category, return None.
        """
        # Special case
        if word.upper() in self.special_words:
            return [word.upper()]
        if word not in self.words_to_pos:
            return None
        return self.words_to_pos[word]

    def get_pos_words(self,pos, meter=None):
        """
        Gets all the words of a given POS
        Parameters
        ----------
        pos - the POS you want
        meter - (optional) returns only words which fit the given meter, e.g. 101
        """
        if pos in self.special_words:
            return [pos.lower()]
        if pos not in self.pos_to_words:
            return None
        if meter:
            ret = [word for word in self.pos_to_words[pos] if word in self.dict_meters and meter in self.dict_meters[word]]
            if len(ret) == 0:
                return False
            return ret
        return self.pos_to_words[pos]

    def write_stanza(self, theme="forest"):
        """
        Possible approach :
        1. Write a preliminary line
            a. fits template and meter randomly and make sure end word has at least 20(?) rhyming words
            b. inserts thematic words where possible(?)
        2. Use bert to change every word (several times?) with weighted probabilities it gives, filtered for meter and template and perhaps relevant words boosted?
        Returns
        -------

        """
        theme_words = [] #TODO
        lines = []
        for template, meter in self.templates:
            template = template.split()
            meter = meter.split("_")
            line = ""
            for i in range(len(template)):
                line += random.choice(self.get_pos_words(template[i], meter=meter[i])) + " "
            print("line initially ", line)
            line = self.update_bert(line.strip().split(), meter, template, 3, verbose=True)
            print("line after ", line)
            lines.append(line)
            break

    def update_bert(self, line, meter, template, iterations, theme_words=[], verbose=False):
        if iterations == 0: return line #base case

        input_ids = torch.tensor(self.tokenizer.encode(line, add_special_tokens=False)).unsqueeze(0) #tokenizes
        outputs = self.bert_model(input_ids)[0][0] #masks each token and gives probability for all tokens in each word. Shape num_words * vocab_size
        softmax = torch.nn.Softmax(dim=1) #normalizes the probabilites to be between 0 and 1
        outputs = softmax(outputs)
        for word_number in range(0,len(line)-1): #ignore  last word to keep rhyme
            if len(self.get_pos_words(template[word_number])) > 1:
                outputs[word_number] += abs(min(outputs[word_number])) #all values are non-negative
                predictions = outputs[word_number].detach().numpy() * [int(word in self.words_to_pos and template[word_number] in self.get_word_pos(word) and word in self.dict_meters and meter[word_number] in self.dict_meters[word]) for word in self.bert_vocab] #filters non-words and words which dont fit meter and template
                #TODO add theme relevance weighting. add internal rhyme and poetic device weighting

                predictions /= sum(predictions)
                if verbose: print("min: ", min(predictions), " max: ", max(predictions), "sum: ", sum(predictions), ", ", predictions)
                line[word_number] = np.random.choice(self.bert_vocab, p=predictions)
                if verbose: print("word now ", line[word_number])

        if verbose: print("line now", line)
        return self.update_bert(line, meter, template, iterations-1, theme_words=theme_words, verbose=verbose)



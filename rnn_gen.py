import os
import json
import sys
import numpy as np
import copy
import random
import scipy
from scipy.stats import hmean
import pickle
import kenlm
from datetime import datetime
import torch

import poem_core
from py_files import helper

from poetry_rnn.verse_generator import VerseGenerator

class Poem(poem_core.Poem):

    def __init__(self, form, config, words_file="saved_objects/tagged_words.p",
                 syllables_file='saved_objects/cmudict-0.7b.txt',
                 extra_stress_file='saved_objects/edwins_extra_stresses.txt',
                 top_file='saved_objects/words/top_words.txt',
                 templates_file="poems/jordan_templates.txt",
                 #templates_file='poems/number_templates.txt',
                 mistakes_file=None):
        poem_core.Poem.__init__(self, words_file=words_file, templates_file=templates_file,
                                syllables_file=syllables_file, extra_stress_file=extra_stress_file, top_file=top_file,
                                mistakes_file=mistakes_file)

        #allow all pronouns
        self.gender = ['i', 'me', 'my', 'mine', 'myself', 'you', 'your', 'yours', 'yourself', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'we', 'us', 'our', 'ours', 'ourselves', 'they', 'them', 'their', 'theirs', 'themselves']


        self.form = form

        self.initializeConfig(config)
        self.loadNMFData()

        self.generator = VerseGenerator(self.MODEL_FILE, self.entropy_threshold)

        self.loadVocabulary()

        self.ngramModel = kenlm.Model(self.NGRAM_FILE)

    def initializeConfig(self, config):
        with open(config) as json_config_file:
            configData = json.load(json_config_file)

        location = os.path.join(
            "/home/home3/ea132/competitors/poetry",
            configData['general']['data_directory'],
            configData['general']['language']
        )

        self.NMF_FILE = os.path.join(location, configData['nmf']['matrix_file'])
        self.NMF_DESCRIPTION_FILE = os.path.join(location, configData['nmf']['description_file'])
        #self.RHYME_FREQ_FILE = os.path.join(location, configData['rhyme']['freq_file'])
        #self.RHYME_DICT_FILE = os.path.join(location, configData['rhyme']['rhyme_dict_file'])
        #self.RHYME_INV_DICT_FILE = os.path.join(location, configData['rhyme']['rhyme_inv_dict_file'])
        self.MODEL_FILE = os.path.join(location, configData['model']['parameter_file'])
        self.NGRAM_FILE = os.path.join(location, configData['model']['ngram_file'])

        self.name = configData['general']['name']
        self.length = configData['poem']['length']
        self.entropy_threshold = configData['poem']['entropy_threshold']

    def loadNMFData(self):
        self.W = np.load(self.NMF_FILE)
        with open(self.NMF_DESCRIPTION_FILE, 'rb') as f:
            self.nmf_descriptions = pickle.load(f, encoding='utf8')


    def loadVocabulary(self):
        self.i2w = self.generator.vocab.itos
        self.w2i = self.generator.vocab.stoi


    def write(self, dim=59, theme=False, ngram=True, use_template=True, use_meter=False, verbose=False):
        self.poem = ""
        self.blacklist_words = set()
        self.blacklist = []
        self.previous_sent = None
        self.filt_meter = use_meter
        if use_template:
            self.writeRhyme(theme, nmfDim=dim, ngram=ngram, verbose=verbose, use_template=self.get_next_words)
        else:
            self.writeRhyme(theme, nmfDim=dim, ngram=ngram, verbose=verbose)
        #print(self.poem)

    def writeRhyme(self, theme, nmfDim="random", ngram=True, use_template=False, verbose=True):

        if nmfDim == 'random':
            nmfDim = random.randint(0,self.W.shape[1] - 1)
        elif type(nmfDim) == int:
            nmfDim = nmfDim
        else:
            nmfDim = None

        if not nmfDim == None:
            sys.stdout.write('\n' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") +' nmfdim ' + str(nmfDim) + ' (' + ', '.join(self.nmf_descriptions[nmfDim]) + ')\n\n')
        else:
            sys.stdout.write('\n' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' NO nmfdim' + '\n\n')

        if nmfDim:
            rhyme_dict = self.getRhymes(self.nmf_descriptions[nmfDim][0], words=self.words_to_pos.keys())
            rhyme_words = []
            for i in range(14):
                r = ""
                if i == 13:
                    while r in rhyme_words or self.w2i[r] == 0:
                        r = random.choice(list(rhyme_dict[rhyme_words[-1]]))
                    rhyme_words.append(r)
                elif i % 4 < 2:
                    while r in rhyme_words or self.w2i[r] == 0 or not any(self.w2i[w] for w in rhyme_dict[r]):
                        if random.random() < 1/(len(rhyme_dict) * 1.5):
                            print("couldnt get rhyme words", i, rhyme_words, rhyme_dict)
                            return 1/0
                        r = random.choice(list(rhyme_dict.keys()))
                    rhyme_words.append(r)
                else:
                    while r in rhyme_words or self.w2i[r] == 0:
                        r = random.choice(list(rhyme_dict[rhyme_words[-2]]))
                    rhyme_words.append(r)
                    if i % 4 == 3: rhyme_words.append("")

        else:
            rhyme_words = ['mountains', 'plains', 'fountains', 'stains', '', 'meadows', 'undergrowth', 'shadows', 'both', '', 'land', 'landscape', 'understand', 'escape', '', 'wild', 'mild']
        sys.stdout.write("\n" + ", ".join(rhyme_words))

        print("writing poem")
        for r in rhyme_words:
            if r:
                try:
                    words = self.getSentence(rhyme_word=r, syllables=True, nmf=nmfDim, ngram=ngram, use_template=use_template, verbose=verbose)
                    words = " ".join(words).replace(" n't", "n't")
                except KeyError as e:
                    print('err', e)
                    continue
                else:
                    sys.stdout.write(words + '\n')
                    self.poem += words + "\n"
                try:
                    self.blacklist_words = self.blacklist_words.union(words)
                except KeyError as e:
                    # means verse does not follow rhyme, probably because of entropy computations
                    # do not show error for presentation
                    # print('err blacklist', e)
                    pass
                except IndexError as e2:
                    print('err blacklist index', e2)
                self.previous_sent = words.split()
            else:
                sys.stdout.write('\n')


    def getSentence(self, rhyme_word, syllables, nmf, use_template, verbose=True, ngram=False):
        if self.previous_sent:
            previous = self.previous_sent
        else:
            previous = None
        if rhyme_word:
            rhymePrior = self.createRhymeProbVector(rhyme_word)
        else:
            rhymePrior = None
        if not nmf == None:
            nmfPrior = copy.deepcopy(self.W[:,nmf])
        else:
            nmfPrior = None

        print(len(self.poem.split("\n")), end=": ")
        allCandidates, allProbScores = self.generator.generateCandidates(previous=previous,rhymePrior=rhymePrior, nmfPrior=nmfPrior, pos_filter=use_template, verbose=verbose)
        if verbose: print("\n\n\ngot the sentences and scores!", len(allCandidates))
        if ngram:
            ngramScores = []
            for ncand, candidate in enumerate(allCandidates):
                try:
                    ngramScore = self.ngramModel.score(' '.join(candidate)) / len(candidate)
                except ZeroDivisionError:
                    ngramScore = -100
                ngramScores.append(ngramScore)
            ngramScores = np.array(ngramScores)
            largest = ngramScores[np.argmax(ngramScores)]
            ngramNorm = np.exp(ngramScores - largest)
        else:
            ngramNorm = np.ones(len(allCandidates))

        allProbScores = np.array([i.cpu().detach().numpy() for i in allProbScores])
        largest = allProbScores[np.argmax(allProbScores)]
        assert largest == allProbScores.max()
        allProbNorm = np.exp(allProbScores - largest)

        if verbose: print("scoring candidates")

        scoreList = []
        for ncand, candidate in enumerate(allCandidates):
            allScores = [allProbNorm[ncand], ngramNorm[ncand]]
            if syllables:
                syllablesScore = self.checkSyllablesScore(candidate, mean=self.length, std=1)
                allScores.append(syllablesScore)
            if nmf:
                NMFScore = self.checkNMF(candidate, [nmf])
                allScores.append(NMFScore)
            allScore = hmean(allScores)
            scoreList.append((allScore, candidate, allScores))

        scoreList.sort()
        scoreList.reverse()

        if verbose: print("got the best", scoreList[:20])

        return scoreList[0][1]


    def createRhymeProbVector(self, rhyme_word):
        probVector = np.empty(len(self.i2w))
        probVector.fill(1e-20)
        probVector[self.w2i[rhyme_word]] = 1
        return probVector / np.sum(probVector)

    def checkSyllablesScore(self, words, mean, std):
        gaussian = scipy.stats.norm(mean,std)
        try:
            nSyllables = sum([len(self.get_meter(w.lower())[0]) if w.lower() in self.dict_meters and len(self.get_meter(w.lower())) else 0 for w in words])
        except:
            print("failed to get number of syllables", words)
            print(1/0)
        return gaussian.pdf(nSyllables) / 0.19

    def checkNMF(self, words, dimList):
        words = list(set([w for w in words if not w in self.blacklist_words]))
        NMFTop = np.max(np.max(self.W[:,dimList], axis=0))
        NMFScore = self.computeNMFScore(words, dimList)
        return NMFScore / NMFTop

    def computeNMFScore(self, words, dimList):
        sm = sum([max(self.W[self.w2i[w], dimList]) for w in words if w in self.w2i])
        return sm


    def get_next_words(self, tokens, verbose=True):
        filt_meter = self.filt_meter
        if type(tokens) == str:
            words = tokens
        else:
            words = " ".join(self.i2w[t] for t in tokens)
        words = words.replace(" n't", "n't").strip("<s>").split()[::-1]
        if verbose: print("words:", words)
        poss_templates = self.get_template_from_line(words, backwards=True)
        if verbose: print("possible templates", len(poss_templates), poss_templates)
        if len(poss_templates) == 0:
            ret = torch.zeros(len(self.i2w))
            if verbose: print("no templates returning", ret, "\n\n\n")
            return ret.tolist()
        [self.check_template(t,m, verbose=False) for t,m in poss_templates]
        n = len(words)
        met = '1010101010'[sum([len(self.get_meter(w)[0]) for w in words if self.get_meter(w)]):]
        try:
            if verbose: print("getting next words for ", helper.remove_punc(poss_templates[0][0].split()[-n - 1]), met)
            next_words = [(helper.remove_punc(t.split()[-n-1]), list(filt_meter * self.get_poss_meters(t.split()[:-n], met).keys())) for t,m in poss_templates]
        except:
            if verbose: print("couldnt find a solution for all")
            next_words = []
            for t,m in poss_templates:
                if len(t.split()) == n:
                    if verbose: print(len(t.split()), "adding eos token")
                    next_words.append(('<s>', []))
                    continue
                pos = helper.remove_punc(t.split()[-n-1])
                if verbose: print(pos, end=" ")
                posmet = self.get_poss_meters(t.split()[:-n], met)
                if verbose: print(posmet)
                if not filt_meter: next_words.append((pos, []))
                elif posmet: next_words.append((pos, list(posmet.keys())))
            #print(1/0)
        if verbose: print("possible words", len(next_words), next_words)
        tokens = set()
        for pos, meter in next_words:
            if not pos:
                continue
            elif pos == '<s>':
                tokens.update([3]) #the eos token
            else:
                tokens.update([self.w2i[w] for w in self.get_pos_words(pos, meter*int(filt_meter)) if self.w2i])

        tokens.discard(0)
        ret = [int(i in tokens) for i in range(len(self.i2w))]
        if verbose: print("returning ret", torch.tensor(ret).size(), torch.tensor(ret).sum(), len(tokens), "\n")
        return ret




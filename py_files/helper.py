import requests
import nltk
from nltk.corpus import wordnet as wn
from gensim.parsing.preprocessing import remove_stopwords
import re
import pickle
import spacy
import numpy as np
spacy_nlp = spacy.load('en_core_web_lg')

def create_syll_dict(fnames, extra_file):
    """
    Using the cmudict file, returns a dictionary mapping words to their
    intonations (represented by 1's and 0's). Assumed to be larger than the
    corpus of words used by the model.

    Parameters
    ----------
    fname : [str]
        The names of the files containing the mapping of words to their
        intonations.
    """
    dict_meters = {}
    for file in fnames:
        with open(file, encoding='UTF-8') as f:
            lines = [line.rstrip("\n").split() for line in f if (";;;" not in line)]

        for i in range(len(lines)):
            line = lines[i]
            newLine = [line[0].lower()]
            if ("(" in newLine[0] and ")" in newLine[0]):
                newLine[0] = newLine[0][:-3]
            chars = ""
            for word in line[1:]:
                for ch in word:
                    if (ch in "012"):
                        if (ch == "2"):
                            chars += "1"
                        else:
                            chars += ch
            newLine += [chars]
            lines[i] = newLine
            if (newLine[0] not in dict_meters):  # THIS IF STATEMENT ALLOWS FOR MULTIPLE PRONUNCIATIONS OF A WORD
                dict_meters[newLine[0]] = [chars]
            else:
                if (chars not in dict_meters[newLine[0]]):
                    dict_meters[newLine[0]] += [chars]
    dict_meters[','] = ['']
    dict_meters['.'] = ['']
    dict_meters["'s"] = ['']

    if extra_file:
        with open(extra_file, "r") as file:
            extras = file.readlines()
            for extra in extras:
                dict_meters[extra.split()[0]].append(extra.split()[1])
    return dict_meters

def create_pos_syllables(pos_to_words, dict_meters):
    """
    Creates a mapping from every pos encountered in the corpus to the all of
    the possible number of syllables across all of the words tagged with
    the given pos.
    """
    pos_syllables = {}
    for k, v in pos_to_words.items():
        pos_syllables[k] = set()
        for w in v:
            try:
                pos_syllables[k].add(len(dict_meters[w][0]))
            except:
                continue
    pos_syllables[','].add(0)
    pos_syllables['.'].add(0)
    return pos_syllables




def get_rhyming_words_one_step_henry(api_url, word, max_syllables=5):
    """
    get the rhyming words of <arg> word returned from datamuse api
    <args>:
    word: any word
    <return>:
    a set of words
    """
    return set(d['word'] for d in requests.get(api_url, params={'rel_rhy': word}).json() if " " not in d['word'] and d['numSyllables'] <= max_syllables)


def get_similar_word_henry(words, seen_words=[], weights=1, n_return=1, word_set=None, api_url='https://api.datamuse.com/words'):
    """
    Given a list of words, return a list of words of a given number most similar to this list.
    <arg>:
    words: a list of words (prompts)
    seen_words: words not to repeat (automatically include words in arg <words> in the following code)
    weights: weights for arg <words>, default to be all equal
    n_return: number of words in the return most similar list
    word_set: a set of words to choose from, default set to the set of words extracted from the definitions of arg <word> in gensim
    <measure of similarity>:
    similarity from gensim squared and weighted sum by <arg> weights
    <return>:
    a list of words of length arg <n_return> most similar to <arg> words
    """
    ps = nltk.stem.PorterStemmer()
    punct = re.compile(r'[^\w\s]')


    seen_words_set = set(seen_words) | set(ps.stem(word) for word in words)

    if word_set is None:
        word_set = set()

        for word in words:
            for synset in wn.synsets(word):
                clean_def = remove_stopwords(punct.sub('', synset.definition()))
                word_set.update(clean_def.lower().split())
            word_set.update({dic["word"] for dic in requests.get(api_url, params={'rel_syn': "grace"}).json()})

    if weights == 1:
        weights = [1] * len(words)

    def cal_score(words, weights, syn):
        score = 0
        for word, weight in zip(words, weights):
            score += max(get_spacy_similarity(word, syn), 0) ** 0.5 * weight
        return score / sum(weights)

    syn_score_list = [(syn, cal_score(words, weights, syn)) for syn in word_set if ps.stem(syn) not in seen_words_set and syn in spacy_nlp.vocab]
    syn_score_list.sort(key=lambda x: x[1], reverse=True)

    return [e[0] for e in syn_score_list[:n_return]]


def get_spacy_similarity(word1, word2):
    if word1 not in spacy_nlp.vocab or word2 not in spacy_nlp.vocab: return 0
    return spacy_nlp(word1).similarity(spacy_nlp(word2))

def isIambic(word):
    #simply return whether or not the word alternates stress ie 1010 or 01010 etc
    for i in range(len(word) - 1):
        if word[i] == word[i+1]:
            return False
    return True

def template_sylls_checking(pos_set, sylls_set, template_curr, num_sylls_curr, possible, num_sylls, pos_sylls_mode): #is this what I want?
    """
    Check whether the current word could fit into our template with given syllables constraint

    Parameters
    ----------
    pos_set: set
        POS of the current word
    sylls_set: set
        Possible number of syllabes of the current word
    template_curr: list
        Partial, unfinished POS template of the current line (e.g. [NN, VB, NN])
    num_sylls_curr: int
        Syllable count of the partially constructed sentence
    possible: list
        All possible POS templates associated with the current line
    num_sylls: int
        predefined number of syllables the current line should have (e.g. 6,9)

    Returns
    -------
    list
        Format is [(POS, sylls)], a combination of possible POS
        and number of syllables of the current word
    """
    continue_flag=set()
    for t in possible:
        if t[:len(template_curr)]==template_curr and len(t)>len(template_curr)+1:
            for pos in pos_set:
                if pos==t[len(template_curr)]:
                    for sylls in sylls_set: #where are there several possible syllables?
                        sylls_up, sylls_lo=sylls_bounds(t[len(template_curr)+1:], pos_sylls_mode)
                        if num_sylls-num_sylls_curr-sylls>=sylls_lo and num_sylls-num_sylls_curr-sylls<=sylls_up:
                            continue_flag.add((pos,sylls))
    if len(continue_flag)==0: continue_flag=False
    return continue_flag


def sylls_bounds(partial_template, pos_sylls_mode):
    """
    Return upper and lower bounds of syllables in a POS template.
    """

    threshold = 0.1
    sylls_up = 0
    sylls_lo = 0
    if len(partial_template) == 1:
        return 2, 1
    else:
        for t in partial_template[:-1]:
            x = [j[0] for j in pos_sylls_mode[t] if j[1] >= min(threshold, pos_sylls_mode[t][0][1])]
            if len(x) == 0:
                sylls_up += 0
                sylls_lo += 0
            else:
                sylls_up += max(x)
                sylls_lo += min(x)
    sylls_up += 2
    sylls_lo += 1
    return sylls_up, sylls_lo

def softmax(x, exclude_zeros=False):
    """Compute softmax values for each sets of scores in x.
       exclude_zeros (bool) retains zero elements
    """
    if exclude_zeros:
        if max(x) <= 0:
            print("max <=0 so retrying without exclusion")
            return softmax(x) #has to be at least one non negative
        elif max(x) == min(x): return np.array(x)/len(x)
        e_x = np.array([int(q > 0) * np.exp(q - np.max(x)) for q in x])
    else:
        e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_pos_dict(postag_file, mistakes_file=None):
    with open(postag_file, 'rb') as f:
        postag_dict = pickle.load(f)
    pos_to_words = postag_dict[1]
    words_to_pos = postag_dict[2]
    if mistakes_file:
        with open(mistakes_file, "r") as pickin:
            list = {l.split()[0]: l.split()[1:] for l in pickin.readlines()}
            for word in list:
                words_to_pos[word] = list[word]
                for pos in pos_to_words:
                    if word in pos_to_words[pos]:
                        pos_to_words[pos].remove(word)
                    if pos in list[word]:
                        pos_to_words[pos].append(word)
    return pos_to_words, words_to_pos

def get_new_pos_dict(file):
    dict = pickle.load(open(file, "rb"))
    words_to_pos = {}
    pos_to_words = {}
    for word in dict:
        if len(word) == 1 and word != "a": continue
        pos = list(dict[word])
        words_to_pos[word] = pos
        for p in pos:
            if p not in pos_to_words: pos_to_words[p] = {}
            pos_to_words[p][word] = 1
    pos_to_words["POS"] = {}
    pos_to_words["POS"]["'s"] = 1
    return pos_to_words, words_to_pos





def get_finer_pos_words():
    return {'WHAT', 'MORE', 'EXCEPT', 'WITHOUT', 'ASIDE', 'WHY',
     'AWAY', 'OF', 'COULD', 'WHOSOEVER', 'WHENEVER', 'SHALL', 'ALBEIT',
     'FROM', 'BETWEEN', 'CAN', 'HOW', 'OUT', 'CANNOT',
     'SO', 'BACK', 'ABOUT', 'LATER', 'IF', 'REGARD',
     'MANY', 'TO', 'THERE', 'UNDER', 'APART',
     'QUITE', 'LIKE', 'WHILE', 'AS', 'WHOSE',
     'AROUND', 'NEITHER', 'WHOM', 'SINCE', 'ABOVE', 'THROUGH', 'ALL',
     'AND', 'SOME', 'MAY', 'HALF', 'WHATEVER', 'BEHIND',
     'BEYOND', 'WHERE', 'SUCH', 'YET', 'UNTO', 'BY', 'NEED',
     'A', 'DURING', 'AT', 'AN', 'OUGHT',
     'BUT', 'DESPITE', 'SHOULD', 'THOSE', 'FOR', 'WHEREVER', 'WHOLE', 'THESE',
     'WHOEVER', 'WITH', 'TOWARD', 'WHICH',
     'BECAUSE', 'WHETHER', 'ONWARD', 'UPON', 'JUST', 'ANY',
     'NOR', 'THROUGHOUT', 'OFF', 'EVERY', 'UP', 'NEXT', 'THAT', 'WOULD',
     'WHATSOEVER', 'AFTER', 'ONTO', 'BESIDE', 'ABOARD', 'OVER', 'BENEATH',
     'INSIDE', 'WHEN', 'OR', 'MUST', 'AMONG', 'MIGHT', 'NEAR', 'PLUS', 'UNTIL',
     'ALONG', 'INTO', 'BOTH', 'EITHER', 'WILL', 'IN',
     'EVER', 'ON', 'AGAINST', 'EACH', 'BELOW',
     'DOWN', 'BEFORE', 'THE', 'WHICHEVER', 'WHO', 'PER', 'THIS',
     'ACROSS', 'THAN', 'WITHIN', 'NOT', "IS", "ARE", "OH", "EVEN", "DO", "BE", "OFT", "TOO"}
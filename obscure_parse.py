from big_phoney import BigPhoney
import re
import pickle
bp = BigPhoney()

#word_file = open("saved_objects/lost_words_phrontristery.txt", "r")
word_file = open("saved_objects/rare_words_phrontristery.txt", "r")
lines = word_file.readlines()
word_file.close()

syll_file = open("saved_objects/ob_syll_dict.txt", "w+")

postag_dict = pickle.load(open("saved_objects/ob_postag_dict.p", "rb"))

ob_pos_to_words = postag_dict[0]
ob_word_to_pos = postag_dict[1]


type_to_pos = {'v':"VB", 'adj':"JJ", 'n':"NN", 'npl':"NNS", 'vz':"VBZ", 'vd':"VBD", 'ving':"VBG"}

def interpret(line):
    #gets and asks me to interpret one syllable words
    word = line.split()[0]
    sylls = bp.phonize(word)
    if len(re.split('[012]', sylls)) > 1:
        print(word, "too long")
        return word + " NONE"
    else:
        return word + " " + input(line)

for line in lines:
    if len(line.split()) > 2: line = interpret(line)
    word, type = line.split()
    if type not in type_to_pos: continue
    pos = type_to_pos[type]

    ob_pos_to_words[pos].append(word)
    ob_word_to_pos[word] = [pos]
    sylls = bp.phonize(word)
    print(word, "-->", sylls)
    syll_file.write(word.upper() + "  " + sylls + "\n")

    if type == 'v' and word[-1] == 'e':
        lines.append(word + 's vz')
        lines.append(word + 'd vd')
        lines.append(word[:-1] + 'ing ving')
        """ ob_pos_to_words['VBZ'].append(word + 's')
        ob_word_to_pos[word + 's'] = ['VBZ']
        sylls = bp.phonize(word + 's')
        print(word + 's', "-->", sylls)
        syll_file.write((word + 's').upper() + "  " + sylls + "\n")
        

        ob_pos_to_words['VBD'].append(word + 'd')
        ob_word_to_pos[word + 'd'] = ['VBD']
        sylls = bp.phonize(word + 'd')
        print(word + 'd', "-->", sylls)
        syll_file.write((word + 'd').upper() + "  " + sylls + "\n")
       

        ob_pos_to_words['VBG'].append(word[:-1] + 'ing')
        ob_word_to_pos[word[:-1] + 'ing'] = ['VBG']
        sylls = bp.phonize(word[:-1] + 'ing')
        print(word[:-1] + 'ing', "-->", sylls)
        syll_file.write((word[:-1] + 'ing').upper() + "  " + sylls + "\n")"""

    elif type == 'n' and word[-1] in ['n', 'r', 's']: #common regular noun endings
        """ob_pos_to_words['NNS'].append(word + 's')
        ob_word_to_pos[word + 's'] = ['NNS']
        sylls = bp.phonize(word + 's')
        print(word + 's', "-->", sylls)
        syll_file.write((word + 's').upper() + "  " + sylls + "\n")"""
        lines.append(word + 's' + "  npl")

syll_file.close()

print(ob_pos_to_words)

import pickle
postag_dict = [ob_pos_to_words, ob_word_to_pos]

pickle.dump(postag_dict, open("saved_objects/ob_postag_dict.p", "wb+"))

from big_phoney import BigPhoney
import random
import pickle
bp = BigPhoney()

#word_file = open("saved_objects/words/fict_words.txt", "r")
#word_file = open("saved_objects/words/lost_words_phrontristery.txt", "r")
word_file = open("saved_objects/words/rare_words_phrontristery.txt", "r")
#word_file = open("saved_objects/words/one_syll_wonders.txt", "r")
lines = word_file.readlines()
word_file.close()

syll_file = open("saved_objects/ob_syll_dict.txt", "a")

one_syll_file = open("saved_objects/words/two_syll_wonders.txt", "a")

type_to_pos = {'v':"VB", 'adj':"JJ", 'n':"NN", 'npl':"NNS", 'vz':"VBZ", 'vd':"VBD", 'ving':"VBG", 'vbp':"VBP", "adv":"RB"}



postag_dict = pickle.load(open("saved_objects/ob_postag_dict.p", "rb"))

ob_pos_to_words = postag_dict[0]
ob_word_to_pos = postag_dict[1]




def interpret(line):
    #gets and asks me to interpret one syllable words
    word = line.split()[0]
    if len(word) > 8 or len(word) < 4:
        print(word, "wrong length")
        return word + " NONE" #assume anything longer than 5 letters cant be 1 syll
    #sylls = bp.phonize(word)
    if bp.count_syllables(word) != 2:
        print(word, "not 2 sylls")
        return word + " NONE"
    else:
        st = word + " " + input(line)
        if st.split()[1] in type_to_pos:
            one_syll_file.write("\n" + st)
        return st

for line in lines:
#while True:
#    line = random.choice(lines)
    if line.split()[0] in ob_word_to_pos: continue
    if len(line.split()) > 2:
        line = interpret(line)
        if line.split()[-1] == "-1":
            one_syll_file.close()
            break
    word, type = line.split()
    if type not in type_to_pos: continue
    pos = type_to_pos[type]

    if pos not in ob_pos_to_words: ob_pos_to_words[pos] = []
    ob_pos_to_words[pos].append(word)
    ob_word_to_pos[word] = [pos]
    sylls = bp.phonize(word)
    """if type == "vz":
        print(word, "-->", sylls.replace("1", "0"))
        syll_file.write(word.upper() + "  " + sylls.replace("1", "0") + "\n")"""
    print(word, "-->", sylls)
    syll_file.write(word.upper() + "  " + sylls + "\n")
    if type == 'v' and word[-1] == 'e':
        lines.append(word + 'vbp')
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

    elif type == 'n' and word[-1] in ['n', 'r', 'm', 't']: #common regular noun endings
        lines.append(word + 's' + "  npl")
        """ob_pos_to_words['NNS'].append(word + 's')
        ob_word_to_pos[word + 's'] = ['NNS']
        sylls = bp.phonize(word + 's')
        print(word + 's', "-->", sylls)
        syll_file.write((word + 's').upper() + "  " + sylls + "\n")"""


syll_file.close()

one_syll_file.close()

print(ob_pos_to_words)

postag_dict = [ob_pos_to_words, ob_word_to_pos]

pickle.dump(postag_dict, open("saved_objects/ob_postag_dict.p", "wb+"))

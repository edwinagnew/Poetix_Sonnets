import pickle
type_to_pos = {'v':"VB", 'adj':"JJ", 'n':"NN", 'npl':"NNS", 'vz':"VBZ", 'vd':"VBD", 'ving':"VBG", 'vbp':"VBP", "adv":"RB"}

slines = open("saved_objects/ob_syll_dict.txt", "r").readlines()
sylls = [s.split()[0].lower() for s in slines]

words = []
for f in ["saved_objects/words/lost_words_phrontristery.txt", "saved_objects/words/one_syll_wonders.txt",
          "saved_objects/words/two_syll_wonders.txt", "saved_objects/words/fict_words.txt", "saved_objects/words/misc_words.txt"]:
    file = open(f, "r")
    words += file.readlines()
    file.close()
print(words)

def reset_postag():
    ob_p2w = {}
    ob_w2p = {}
    for l in words:
        word, pos = l.split()
        if pos not in type_to_pos:
            print("help", l)
            print(1/0)
        pos = type_to_pos[pos]
        if word in sylls:
            if pos not in ob_p2w: ob_p2w[pos] = []
            ob_p2w[pos].append(word)
            ob_w2p[word] = [pos]

            if pos == 'v' and word[-1] == 'e':
                words.append(word + ' vbp')
                words.append(word + 's vz')
                words.append(word + 'd vd')
                words.append(word[:-1] + 'ing ving')
            elif pos == 'n' and word[-1] in ['n', 'r', 'm', 't']:  # common regular noun endings
                words.append(word + 's' + "  npl")
        else:
            print(word, "not in sylls")
    return [ob_p2w, ob_w2p]

if input("reset postag?") == "y":
    update = reset_postag()
    print(update[0])
    print(update[1])
    if input("save?") == "y": pickle.dump(update, open("saved_objects/ob_postag_dict.p", "wb"))


def match_sylls():
    known = {w.split()[0]: w.split()[1] for w in words}
    misc_file = open("saved_objects/words/misc_words.txt", "a")
    for s in sylls:
        if s not in known:
            if s[:-1] in known:
                if s[-1] == "d":
                    print(s + " vd\n")
                    misc_file.write(s + " vd\n")
                elif s[-2:] == "es":
                    print(s + " vz\n")
                    misc_file.write(s + " vz\n")
                elif s[-1] == "s":
                    print(s + " npl\n")
                    misc_file.write(s + " npl\n")
            elif s[-3:] == "ing":
                print(s + " ving\n")
                misc_file.write(s + " ving\n")
            else:
                new = s + " " + input(s) + "\n"
                print(new)
                misc_file.write(new)
    misc_file.close()

if input("update fict_words") == "y":
    match_sylls()
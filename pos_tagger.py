import nltk
#nltk.download()
import pickle
import string

import sonnet_basic
s = sonnet_basic.Sonnet_Gen()

f = open("poems/shakespeare_all_sonnets.txt", "r")
lines = f.readlines()
print(len(lines))
print(lines[-3:])

pos_file = open("poems/shakespeare_templates.txt", "a")
couplets = []
end_pos = {"NN"}

archaic = {"thou": "PRP", "thine":"NNS", "thy":"PRP$", "thee":"PRP"}

begin = int(input("What line to start?"))
for i in range(begin, len(lines) - 1, 2): #deal with punctuation?
    if len(lines[i]) < 2:
        print(lines[i])
        continue

    line1 = nltk.word_tokenize(lines[i].lower().translate(str.maketrans('', '', string.punctuation)))
    line2 = nltk.word_tokenize(lines[i+1].lower().translate(str.maketrans('', '', string.punctuation)))

    tags1 = nltk.pos_tag(line1) #doesnt work very well apparently
    tags2 = nltk.pos_tag(line2)
    print(lines[i])
    wds = lines[i].split()
    for x in range(len(wds)):
        if wds[x].upper() in s.special_words:
            tags1[x] = (wds[x].lower(), s.get_word_pos(wds[x].lower())[0])
        if wds[x].lower() in archaic:
            tags1[x] = (wds[x].lower(), archaic[wds[x].lower()])



    if input("How is " + str(tags1)) == "1":
        k = [t[1] for t in tags1]
    else:
        wrong = input("Which word: " + str([(j, tags1[j], s.get_word_pos(tags1[j][0])) for j in range(len(tags1))])).split()
        k = [t[1] for t in tags1]
        for w in wrong:
            w = int(w)
            st = "which POS for " + str(tags1[w]) + str(s.get_word_pos(tags1[w][0])) + " then? "
            k[w] = input(st)

    print("ok", k)
    print(lines[i])
    """ write = input("write to text file?")
    if write:
        print("writing")
        pos_file.write(str(k))"""
    print("")
    print(lines[i+1])
    wds = lines[i+1].split()
    for x in range(len(wds)):
        if wds[x].upper() in s.special_words:
            tags2[x] = (wds[x].lower(), s.get_word_pos(wds[x].lower())[0])
        if wds[x].lower() in archaic:
            tags2[x] = (wds[x].lower(), archaic[wds[x].lower()])

    if input(tags2) == "1":
        v = [t[1] for t in tags2]
    else:
        wrong = input("Which word: " + str([(j, tags2[j], s.get_word_pos(tags2[j][0])) for j in range(len(tags2))])).split()
        v = [t[1] for t in tags2]
        for w in wrong:
            w = int(w)
            st = "which POS for " + str(tags2[w]) + str(s.get_word_pos(tags2[w][0])) + " then? "
            v[w] = input(st)

    print("ok", v)
    print(lines[i+1])
    """write = input("write to text file?")
    if write:
        print("writing")
        pos_file.write(str(k))"""
    print("")

    #couplets.append([k,v]) one day do this, but for now is kind of complicated or just broken
    couplets.append(k)
    couplets.append(v)


    end_pos.add(k[-1])
    end_pos.add(v[-1])


print(len(couplets))

pickle.dump(couplets, open("poems/shakespeare_tagged_verified.p", "wb"))

#pickle.dump(end_pos, open("poems/end_pos.p", "wb"))

pos_file.close()
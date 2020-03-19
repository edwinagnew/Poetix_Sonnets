import nltk
#nltk.download()
import pickle
import sys

import sonnet_basic
s = sonnet_basic.Sonnet_Gen()

f = open("poems/shakespeare_all_sonnets.txt", "r")
lines = f.readlines()

pos_file = open("poems/shakespeare_templates.txt", "a")
couplets = []
end_pos = {"NN 1"}

archaic = {"thou": "PRP", "thine": "PRP$", "thy": "PRP$", "thyself": "PRP", "thee": "PRP", "art": "VBP", "doth": "VBZ", "dost": "VBP"}
if len(sys.argv) > 1:
    begin = int(sys.argv[-1])
else:
    begin = int(input("What line to start?"))
for i in range(begin, len(lines)): #deal with punctuation?
    if len(lines[i]) < 2:
        print(lines[i])
        continue

    line1 = nltk.word_tokenize(lines[i].lower())#.translate(str.maketrans('', '', string.punctuation)))

    tags1 = nltk.pos_tag(line1) #doesnt work very well apparently
    print(lines[i])
    wds = lines[i].split()
    meter = ""
    for x in range(len(wds)):
        if wds[x].upper() in s.special_words:
            tags1[x] = (wds[x].lower(), s.get_word_pos(wds[x].lower())[0])
        if wds[x].lower() in archaic:
            tags1[x] = (wds[x].lower(), archaic[wds[x].lower()])
        if wds[x].lower() in s.dict_meters:
            m = s.dict_meters[wds[x].lower()]
            if x == 0 and len(m[0]) == 1 and len([q for q in m if q[0] == "0"]) == 0: #if the first word is one syllable and never 0, make it so
                print("adding", wds[x].lower(), " 0")
                with open("saved_objects/edwins_extra_stresses.txt", "a") as extras_file:
                    extras_file.write("\n" + wds[x].lower() + " 0")
                extras_file.close()
                m.append("0")
            if len(m) > 1:
                for mi in m:
                    if (len(meter) == 0 and mi[0] == "0") or (len(meter) != 0 and meter[-2] != mi[0]):
                        meter += mi + "_"
                        break
            else:
                meter += m[0] + "_"
        else:
            meter += "IDK_"
    meter = meter[:-1] #remove last _


    if input("How is " + str(tags1) + meter) == "1":
        k = [t[1] for t in tags1]
    else:
        print("")
        wrong = input("Which word: " + str([(j, tags1[j], s.get_word_pos(tags1[j][0])) for j in range(len(tags1))])).split()
        k = [t[1] for t in tags1]
        for w in wrong:
            w = int(w)
            st = "which POS for " + str(tags1[w]) + str(s.get_word_pos(tags1[w][0])) + " then? "
            k[w] = input(st)

    print("ok", k, meter)
    print(lines[i])
    print(i)
    """ write = input("write to text file?")
    if write:
        print("writing")
        pos_file.write(str(k))"""




    #couplets.append([k,v]) one day do this, but for now is kind of complicated or just broken
    couplets.append(k)



    end_pos.add(k[-1] + " " + str(meter[-1]))
    print("end_pos ", k[-1] , " " , len(meter.split("_")[-2]))

    print("")



print(len(couplets))

#pickle.dump(couplets, open("poems/shakespeare_tagged_verified.p", "wb"))

#pickle.dump(end_pos, open("poems/end_pos.p", "wb"))

pos_file.close()
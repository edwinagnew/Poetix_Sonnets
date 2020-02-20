import nltk
#nltk.download()
import pickle
import string


f = open("poems/shakespeare_all_sonnets.txt", "r")
lines = f.readlines()
print(len(lines))
print(lines[-3:])
couplets = []
end_pos = {"NN"}
for i in range(0, len(lines) - 1, 2): #deal with punctuation?
    if len(lines[i]) < 2:
        print(lines[i])
        continue

    line1 = nltk.word_tokenize(lines[i].translate(str.maketrans('', '', string.punctuation)))
    line2 = nltk.word_tokenize(lines[i+1].translate(str.maketrans('', '', string.punctuation)))

    tags1 = nltk.pos_tag(line1) #doesnt work very well apparently
    tags2 = nltk.pos_tag(line2)

    k = [t[1] for t in tags1]
    v = [t[1] for t in tags2]

    #couplets.append([k,v]) one day do this, but for now is kind of complicated or just broken
    couplets.append(k)
    couplets.append(v)
    if i > 2130: print(i, v, lines[i+1])

    end_pos.add(k[-1])
    end_pos.add(v[-1])


print(len(couplets))

pickle.dump(couplets, open("poems/shakespeare_tagged.p", "wb"))

pickle.dump(end_pos, open("poems/end_pos.p", "wb"))
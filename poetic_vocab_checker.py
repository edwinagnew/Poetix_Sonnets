print("loading...")
import sonnet_basic
s = sonnet_basic.Sonnet_Gen()

from nltk import PorterStemmer
stemmer = PorterStemmer()

words_file = open("saved_objects/words/more_poetic_words.txt", "r").readlines()

words = {}
for line in words_file:
    w, *z, n = line.split()
    if "[" in w or "]" in w: continue
    if w.lower() not in words: words[w.lower()] = 0
    words[w.lower()] += int(n)


extra_file = open("saved_objects/words/poetic_word_pos.txt", "a")
a = open("saved_objects/words/poetic_word_pos.txt", "r").readlines()
already_checked = {w.split()[0]: w.split()[1:] for w in a}

print("you've got", len([w for w in words if w not in already_checked and (w not in s.dict_meters or w not in s.words_to_pos)]), "to check :)")
print("loaded. type quit when it asks for pos to quit")
print("type nothing to delete any word")
for word in words:
    if word not in already_checked:
        met = ""
        pos = ""
        while word not in s.dict_meters:
            if stemmer.stem(word) in s.dict_meters:
                print("NB -", stemmer.stem(word), " is ", s.dict_meters[stemmer.stem(word)])
            met = input("what is the meter for '" + word + "' ? e.g. 10 (type a word to see its meter to compare if unsure) : ")
            if met == "quit":
                break
            elif not met:
                print("deleting", word)
                break
            elif met[0] not in ["0", "1"]:
                if met in s.dict_meters:
                    print(met," is ", s.dict_meters[met])
                else:
                    print(met, "not known")
            else:
                s.dict_meters[word] = met


        if not s.get_word_pos(word) and word in s.dict_meters:
            pos = input("what is the pos for '" + word + "' ? (seperate with spaces if multiple): ")
            if pos == "quit":
                break
            s.words_to_pos[word] = {pos: words[word]}
        if len(met + pos) > 1:
            extra_file.write(word + " " + pos + " " + met + "\n")

print("saving and quitting ...")
extra_file.close()


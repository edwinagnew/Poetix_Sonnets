print("loading...")
import sonnet_basic
s = sonnet_basic.Sonnet_Gen()

from nltk import PorterStemmer
stemmer = PorterStemmer()

words_file = open("saved_objects/words/bryon_words.txt", "r").readlines()

extra_file = open("saved_objects/words/byron_pos.txt", "a")
a = open("saved_objects/words/byron_pos.txt", "r").readlines()
already_checked = {w.split()[0]: w.split()[1:] for w in a}

print("loaded. type quit at any point to quit")
for word in words_file:
    word = word.split(" : ")[0].lower()
    if word not in already_checked:
        met = ""
        pos = ""
        while word not in s.dict_meters:
            if stemmer.stem(word) in s.dict_meters:
                print("NB -", stemmer.stem(word), " is ", s.dict_meters[stemmer.stem(word)])
            met = input("what is the meter for '" + word + "' ? e.g. 10 (type a word to see its meter to compare if unsure) : ")
            if met == "quit":
                break
            elif met[0] not in ["0", "1"]:
                if met in s.dict_meters:
                    print(met ," is ", s.dict_meters[met])
                else:
                    print(met, "not known")
            else:
                s.dict_meters[word] = met

        if not s.get_word_pos(word):
            pos = input("what is the pos for '" + word + "' ? (seperate with spaces if multiple): ")
            if pos == "quit":
                break
        if len(met + pos) > 1:
            extra_file.write(word + " " + pos + " " + met + "\n")

print("saving and quitting ...")
extra_file.close()


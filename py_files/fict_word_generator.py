import fictionary
#from big_phoney import BigPhoney
#bp = BigPhoney()


fict_file = open("saved_objects/words/fict_words.txt", "a")
vowels = ["a", "e", "i", "o", "u"]
while True:
    word = fictionary.word(4,10)
    if word[-2:] == "ed":
        pos = "vd"
    elif word[-1] == "s" and word[-2] not in ["s"] + vowels:
        pos = "npl"
    elif word[-3:] == "ing":
        pos = "ving"
    elif word[-2:] == "ly":
        pos = "adv"
    else:
        pos = input(word+ " ")
        if pos == "-1": break
    if pos != "no":
        print(word + " -> " + pos)
        fict_file.write(word + " " + pos + "\n")

fict_file.close()






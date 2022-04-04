import poem_core

if __name__ == '__main__':

    #file1 = open("saved_objects/jj_mistakes.txt", "w")
    s = poem_core.Poem()
    words = [item + " \n" for item in list(s.pos_to_words["JJ"].keys())]

    print(len(words))



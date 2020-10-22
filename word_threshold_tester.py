import scenery




def test_diff(diffs):
    poem1 = scenery.Scenery_Gen()
    poem2 = scenery.Scenery_Gen()
    for num1, num2 in diffs:
        #for theme in ["love war", "peace hate", "snow forest", "dark forest", "youth dreams"]:
        for theme in ["love war"]:
            poem1.pos_to_words = poem1.vocab_orig.copy()
            poem2.pos_to_words = poem2.vocab_orig.copy()
            for p in ["NN", "NNS", "ABNN"]:
                print("theme: ", theme)
                poem1.pos_to_words[p] = {word: s for (word, s) in poem1.pos_to_words[p].items() if
                                        poem1.fasttext.word_similarity(word, theme.split()) > num1}
                poem2.pos_to_words[p] = {word: s for (word, s) in poem1.pos_to_words[p].items() if
                                         poem1.fasttext.word_similarity(word, theme.split()) > num2}
                print("Words cut from %s with a threshold of %f that are kept with a threshold of %f" % (p,num2,num1))
                print(set(poem1.pos_to_words[p]) - set(poem2.pos_to_words[p]))
                print("length of set1 : ", len(set(poem1.pos_to_words[p])))
                print("length of set2 : ", len(set(poem2.pos_to_words[p])))
                print()


def run_diffs():
    test_diff([(.3, .35)])
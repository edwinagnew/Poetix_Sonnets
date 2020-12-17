def make_word_set(file):
    f = open(file)
    word_set = set()
    for line in f:
        word_set = word_set.union(set([word.lower() for word in line.strip().split()]))
    return word_set
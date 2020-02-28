
class Line:

    def __init__(self, text, score, dict_meters, template=None):
        self.text = text
        self.score = score
        self.template = template
        self.syllables = 0
        self.is_iambic = False
        self.update()

        self.dict_meters = dict_meters
        print(self.dict_meters)


    def update(self):
        sylls = 0
        iambic = True
        for word in self.text.split():
            meter = self.dict_meters[word][0]
            sylls += len(meter)
            iambic *= self.isIambic(word)

        self.is_iambic = iambic
        self.syllables = sylls

    def isIambic(self, word):
        # simply return whether or not the word alternates stress ie 1010 or 01010 etc
        for i in range(len(word) - 1):
            if word[i] == word[i + 1]:
                return False
        return True




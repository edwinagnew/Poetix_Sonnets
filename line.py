
class Line:

    def __init__(self, text, dict_meters, score=None, template=None):
        self.text = text
        self.score = score
        self.template = template
        self.dict_meters = dict_meters
        self.meter = ""
        self.syllables = 0
        self.update()


    def update(self):
        self.meter = ""
        self.syllables = 0
        for word in self.text.split():
            meter = self.dict_meters[word][0]
            self.syllables += len(meter)
            self.meter += meter

    def add_word(self, new_word, sylls):
        self.text = new_word + " " + self.text
        self.meter = sylls + self.meter
        self.syllables += len(sylls)


    def print_info(self, text=True, template=True, meter=True):
        if text: print("TEXT:", self.text)
        if template: print("TEMPLATE:", self.template)
        if meter: print("METER:", self.meter)
        print("")

    def reset(self):
        self.text = self.text.split()[-1].strip()
        self.template = self.template.split()[-1].strip()
        self.update()






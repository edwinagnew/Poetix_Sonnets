
class Line:

    def __init__(self, text, meter, pos_template=None):
        self.text = ""
        #self.score = score
        self.pos_template = pos_template
        self.meter = ""
        self.syllables = 0
        self.add_word(text, meter)
        #self.update()


    def add_word(self, new_word, sylls):
        if self.syllables == 0:
            self.text = new_word
            self.meter = sylls
        else:
            self.text = new_word + " " + self.text
            self.meter = sylls + "_" + self.meter
        self.syllables += len(sylls)


    def print_info(self, text=True, pos_template=True, meter=True):
        if text: print("TEXT:", self.text)
        if pos_template: print("TEMPLATE:", self.pos_template)
        if meter: print("METER:", self.meter)
        print("")

    def reset(self):
        self.text = self.text.split()[-1].strip()
        self.pos_template = self.pos_template.split()[-1].strip()
        self.meter = self.meter.split('_')[-1].strip()
        self.syllables = len(self.meter)
        print("RESET")
        self.print_info()







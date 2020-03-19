
class Line:

    def __init__(self, text, dict_meters, pos_template=None):
        self.text = text
        #self.score = score
        self.pos_template = pos_template
        #self.met_template = met_template
        self.dict_meters = dict_meters
        self.meter = ""
        self.syllables = 0
        #self.update()


    """def update(self):
        self.meter = ""
        self.syllables = 0
        words = self.text.split()
        for i in range(len(words)):
            meter = [poss_meter for poss_meter in self.dict_meters[words[i]] if poss_meter in METER TEMPLATE AT RIGHT POSITION]
            self.syllables += len(meter)
            self.meter += meter"""

    def add_word(self, new_word, sylls):
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







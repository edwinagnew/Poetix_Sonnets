class Sentence():

    def __init__(self, encodes, score, text, template, current_line_template, how_many_syllabus_used_in_current_line, rhyme, moving_average):
        self.encodes = encodes
        self.score = score
        self.text = text
        self.template = template
        self.current_template = current_line_template
        self.line_syllables = how_many_syllabus_used_in_current_line
        self.rhyme = rhyme
        self.moving_average = moving_average

#import scenery
import poem_core
import pickle
import random
import gpt_revised

class Succession_Trainer:
    def __init__(self, scenery_object, matrix_file="saved_objects/template_matrix.p"):


        self.scenery_object = scenery_object
        try:
            self.matrix_dict = pickle.load(open(matrix_file, "rb"))
        except:
            templates = []
            for file in ["templates_present", "templates_past", "templates_future", "templates_basic"]:
                file_path = "poems/" + file + ".txt"
                f = open(file_path, "r")
                for line in f.readlines():
                    templates.append(" ".join(line.split()[:-1])) #to get rid of the meter

            self.matrix_dict = {template: {} for template in templates}

        self.line_gen = gpt_revised.Line_Generator(sonnet_object=scenery_object, gpt_object=scenery_object.gpt)

    def update_matrix(self, couplet, template_a, template_b):
        score = self.scenery_object.gpt.score_line(couplet)
        if template_a not in self.matrix_dict:
            self.matrix_dict[template_a] = {}
        if template_b not in self.matrix_dict[template_a]:
            self.matrix_dict[template_a][template_b] = []
        self.matrix_dict[template_a][template_b].append(score)


        return

    def save_matrix_dict(self, filepath="saved_objects/template_matrix.p"):
        pickle.dump(self.matrix_dict, open(filepath, "wb"))
        return

    def gen_couplet(self, t1=None):
        if t1 is None:
            (t1, m1) = random.choice(self.scenery_object.templates)
        else:
            m1 = self.scenery_object.all_templates_dict[t1]

        m1_dict = self.scenery_object.get_meter_dict(t1, m1)
        if not m1_dict:
            self.line_gen.reset()
            return "fail1"


        self.line_gen.new_line(t1, m1_dict)
        line1 = self.line_gen.complete_lines()[t1][0].curr_line.capitalize()

        t2, m2 = self.scenery_object.get_next_template([t1])
        m2_dict = self.scenery_object.get_meter_dict(t2, m2)

        self.line_gen.prev_lines = line1
        if not m2_dict:
            self.line_gen.reset()
            return "fail2"
        self.line_gen.new_line(t2, m2_dict)

        line2 = self.line_gen.complete_lines()[t2][0].curr_line

        self.line_gen.reset()

        coup = line1 + "\n" + line2

        self.update_matrix(coup, t1, t2)

        return coup



    def gen_all(self, k=10, verbose=False):
        for j, template in enumerate(self.matrix_dict):
            if verbose: print("\n\n", j, "for", template)
            for i in range(k):
                self.scenery_object.reset_gender()
                c = self.gen_couplet(template)
                if verbose: print("generated:", c)
            self.save_matrix_dict()



        self.save_matrix_dict()


if __name__ == "__main__":
    #sc = scenery.Scenery_Gen()
    p_c = poem_core.Poem()

    gpt = gpt_revised.gpt_gen(p_c)

    p_c.gpt = gpt

    tt = Succession_Trainer(p_c)

    tt.gen_all(verbose=True)

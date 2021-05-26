import scenery
import pickle
class succession_trainer:
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

    def update_matrix(self, couplet, template_a, template_b):
        score = self.scenery_object.gpt.score_line(couplet)
        if template_a not in self.matrix_dict:
            self.matrix_dict[template_a] = {}
        if template_b not in self.matrix_dict[template_a]:
            self.matrix_dict[template_a][template_b] = []
        self.matrix_dict[template_a][template_b].append(score)
        return

    def save_matrix_dict(self, filepath):
        pickle.dump(self.matrix_dict, open(filepath, "wb"))
        return
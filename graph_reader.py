import random
import numpy as np

class Node():
    def __init__(self, word_info=None, word_pos=None, templates=None):
        self.word = word_info
        self.pos = word_pos
        self.edges = {}
        print(templates)
        if templates != None:
            self.templates = [(" ".join(line.split()[:-1]), line.split()[-1]) for line in templates if "#" not in line and len(line) > 1]


class Graph():
    def __init__(self, txt_file="saved_objects/story_graphs/love.txt"):
        self.nodes = []
        f = open(txt_file)
        curr_line = f.readline().strip()
        self.starts = [int(num) for num in curr_line.split(',')]
        curr_line = f.readline().strip().split('=')
        while curr_line[0]!= "":
            if curr_line[0][0] in "0123456789":
                self.nodes[int(curr_line[0])].edges[curr_line[2]] = self.nodes[int(curr_line[1])]
            else:
                curr_line = self.format_for_templates(curr_line)
                self.nodes.append(Node(word_info=curr_line[0],word_pos=curr_line[1],templates=curr_line[2]))
            curr_line = f.readline().strip().split('=')
        self.curr = self.nodes[random.choice(self.starts)]


    def update(self, edges=None, probs=None):
        if edges == None:
            vb = random.choice(list(self.curr.edges.keys()))
        else:
            vb = np.random.choice(list(self.curr.edges.keys()), p=probs)
        self.curr = self.curr.edges[vb]
        return (vb, self.curr.word, self.curr.pos)

    def get_verbs(self):
        return list(self.curr.edges.keys())

    def format_for_templates(self, temp_list):
        """reformats so that the templates are in an iterable format"""
        templates = [item for item in temp_list[2:]]
        return [temp_list[0], temp_list[1], templates]

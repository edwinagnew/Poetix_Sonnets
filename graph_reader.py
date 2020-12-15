
class Node():
    def __init__(self, word_info=None, word_pos=None):
        self.word = word_info
        self.pos = word_pos
        self.edges = {}


class Graph():
    def __init__(self, txt_file="saved_objects/story_graphs/love.txt"):
        self.nodes = []
        f = open(txt_file)
        curr_line = f.readline().strip()
        self.starts = [int(num) for num in curr_line.split(',')]
        curr_line = f.readline().strip().split(',')
        while curr_line[0] != "":
            if curr_line[0] in "0123456789":
                self.nodes[int(curr_line[0])].edges[curr_line[2]] = self.nodes[int(curr_line[1])]
            else:
                self.nodes.append(Node(word_info=curr_line[0],word_pos=curr_line[1]))
            curr_line = f.readline().strip().split(',')

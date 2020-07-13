
import poem_core

if __name__ == '__main__':

    poem = poem_core.Poem()
    file = open("saved_objects/jjs_mistakes.txt", "w")
    to_write = poem.pos_to_words["JJS"]
    file.writelines("\n".join(to_write))
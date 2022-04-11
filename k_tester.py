import scenery
import gpt_2

if __name__ == "__main__":
    s = scenery.Scenery_Gen()
    """
    for theme in ['forest']:
        print("Writing poem for theme '", theme, "' with k=", 3)
        for j in range(5):
            poem = s.write_poem_flex(theme=theme, k=3)
            print(poem)
    """

    for i in range(2):
        for theme in ["war", "forest", "wisdom", "death", "darkness"]:
            print(s.write_poem_flex(theme=theme, k=1))
import scenery


if __name__ == "__main__":
    s = scenery.Scenery_Gen()
    for theme in ['storms', "darkness", "war", "love"]:
        print("Writing poem for theme '", theme, "' with k=", 3)
        for j in range(5):
            poem = s.write_poem_flex(theme=theme, k=3)
            print(poem)
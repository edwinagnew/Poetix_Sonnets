import scenery


if __name__ == "__main__":
    s = scenery.Scenery_Gen()
    for theme in ['war', 'love', 'darkness', 'wisdom', 'death']:
        print("Writing poem for theme '", theme, "' with k=", 10)
        for j in range(2):
            poem = s.write_poem_flex(theme=theme, k=10)
            print(poem)
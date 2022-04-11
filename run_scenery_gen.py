import scenery

if __name__ == "__main__":
    s = scenery.Scenery_Gen()
    print(s.write_poem_revised(k=2, theme="forest", verbose=True))
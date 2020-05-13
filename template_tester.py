print("please wait")

import sonnet_basic
import random

sonnet = sonnet_basic.Sonnet_Gen()

rating_file = open("poems/template_scores.txt", "a")


##Change this to use a different file
path = "poems/shakespeare_templates.txt"
##
def generate_line(template, meter):
    line = ""
    for i in range(len(template)):
        line += random.choice(sonnet.get_pos_words(template[i], meter=meter[i])) + " "
    return line
with open(path, "r") as templs:
    templates = []
    lines = templs.readlines()
    for line in lines:
        templates.append((" ".join(line.split()[:-1]), line.split()[-1].strip()))

print("ok loaded")
print("Type 'quit' when it asks for number, otherwise will keep going forever")

num = input("What template would you like to check? Pick a number between 0 and " + str(len(templates) - 1)  + ": ")

while num != "quit":
    temp = templates[int(num)]
    print("testing: ", temp)
    for i in range(5):
        line = generate_line(temp[0].split(), temp[1].split('_'))
        print(i + 1, "- ", line)
    rating = input("On a scale of 1-10 how good is that template? If you want to see more lines type 'more': ")
    if rating != "more":
        misc = input("any other comments? (just press enter for no) : ")
        rating_file.write(num + "\t" + str(temp[0])+ "\t" + str(temp[1]) + "\t" + rating + "\t" + misc + "\n")
        print("")
        num = input("What template would you like to check next? Pick a number between 0 and " + str(len(templates) - 1) + ": ")

print("saving and quitting...")

rating_file.close()
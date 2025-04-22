import json


file = "test.txt"
dict = {}
with open(file,"r") as f:
    for line in f:
        place, name = line.strip().split(",")

        if place not in dict:
            dict[place] = [name]
        else:
            dict[place].append(name) 


with open("test.json", "w") as f:
    json.dump(dict,f,indent=4)
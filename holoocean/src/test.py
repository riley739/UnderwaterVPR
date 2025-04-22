import json


with open("test.json", "w+") as f:
    json.dump({"frames": []}, f, indent=4)
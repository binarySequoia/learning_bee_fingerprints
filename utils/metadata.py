import json

class Metadata():
    def __init__(self):
        self.metadata = dict()
    
    def add_metadata(self, key, value):
        self.metadata[key]= value
       
    def read_data_metadata(self, fn, key=""):
        data = dict()
        with open(fn, "r") as f:
            data = json.load(f)
        
        if key != "":
            self.metadata[key] = data
        else:
            self.metadata = data
       
    def save(self, fn):
        with open(fn, "w") as f:
            f.write(json.dumps(self.metadata, indent=2))
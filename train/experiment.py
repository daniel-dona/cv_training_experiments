import os


class Experiment:
    
    def __init__(self, name, folder, parameters, model):
        self.name = name
        self.folder = folder
        
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder, exist_ok=True)

        self.parameters = parameters
        self.model = model

    def save(self):
        pass
import yaml

class Config:
    def __init__(self):
        with open("params.yaml", "r") as f:
            self.params = yaml.safe_load(f)
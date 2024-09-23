import litserve as ls
import pickle
import numpy as np
from src.config_reader import Config
from src.request_validator import InferenceRequest

conf = Config()

class InferenceAPI(ls.LitAPI):
    def setup(self, device = "cpu"):
        with open(conf.params['evaluate']['model_path'], "rb") as f:
            self.model = pickle.load(f)
    
    def decode_request(self, request):
        try:
            InferenceRequest(**request["input"])
            data = [val for val in request["input"].values()]
            x = np.asarray(data)
            x = np.expand_dims(x, 0)
            return x
        except:
            return None
    
    def predict(self, x):
        if x is not None:
            return self.model.predict(x)
        else:
            return -1

    def encode_response(self, output):
        if output==-1:
            message = "Error Occurred"
        else:
            message = "Response Produced Successfully"
        return {
            "message": message,
            "prediction": int(output)
        }
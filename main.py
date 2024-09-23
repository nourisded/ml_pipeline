from src.inference_api import InferenceAPI
import litserve as ls

if __name__ == "__main__":
    api = InferenceAPI()
    server = ls.LitServer(api)
    server.run(port=8000)
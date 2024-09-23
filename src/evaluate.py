from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config_reader import Config
import pandas as pd
import json
import pickle

conf = Config()

val_data = pd.read_csv(conf.params['evaluate']['eval_data_path'])
X_val, y_val = val_data.drop("price_range", axis = 1), val_data["price_range"]
def eval_data(y, y_hat) -> None:
    accuracy = accuracy_score(y, y_hat)
    precision = precision_score(y, y_hat, average='macro')
    recall = recall_score(y, y_hat, average='macro')
    f1 = f1_score(y, y_hat, average='macro')
    evals = {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }
    with open(conf.params['evaluate']['validation_file'], "w", encoding="utf-8") as js:
        json.dump(evals, js, indent=2)

if __name__ == "__main__":
    # Load the model
    with open(conf.params['evaluate']['model_path'], "rb") as f:
        model = pickle.load(f)
    preds = model.predict(X_val)
    eval_data(y_val, preds)
from sklearn.ensemble import GradientBoostingClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config_reader import Config
import pandas as pd
import pickle
import mlflow
import dagshub

conf = Config()
dagshub.init(
    repo_owner=conf.params['train']['username'], 
    repo_name=conf.params['train']['repo'], 
    mlflow=True
)

train_data = pd.read_csv(conf.params['train']['train_data_path'])
val_data = pd.read_csv(conf.params['train']['eval_data_path'])
X, y = train_data.drop("price_range", axis = 1), train_data["price_range"]
X_val, y_val = val_data.drop("price_range", axis = 1), val_data["price_range"]

def trainer(ss):
    with mlflow.start_run():
        mlflow.set_tag("developer", "Ezzaldin")
        mlflow.set_tag("model", "GBM")
        mlflow.log_params(ss)
        booster = GradientBoostingClassifier(**ss)
        booster.fit(X, y)
        y_pred = booster.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='macro')
        recall = recall_score(y_val, y_pred, average='macro')
        f1 = f1_score(y_val, y_pred, average='macro')
        mlflow.log_metrics(
            {
                "accuracy": accuracy,
                "f1": f1,
                "precision": precision,
                "recall": recall
            }
        )
        return {'loss': -1*f1, 'status': STATUS_OK}

if __name__ == "__main__":
    SEARCH_SPACE={
        "n_estimators":scope.int(hp.quniform("n_estimators", 20, 700, 5)),
        "max_depth":scope.int(hp.quniform("max_depth", 1, 12, 1)),
        "min_samples_split":scope.int(hp.quniform("num_leaves", 100, 150, 5)),
        "min_samples_leaf":scope.int(hp.quniform("min_samples_leaf", 30, 500, 20)),
        "learning_rate":scope.float(hp.quniform("learning_rate", 0.01, 0.3, 0.001))
    }
    trails = Trials()
    best_res = fmin(
        fn = trainer,
        space = SEARCH_SPACE,
        algo = tpe.suggest,
        max_evals=5,
        trials=trails
    )
    params = {
        'learning_rate': best_res['learning_rate'],
        'max_depth': int(best_res['max_depth']),
        'min_samples_leaf': int(best_res['min_samples_leaf']),
        'n_estimators': int(best_res['n_estimators']),
        'min_samples_split': int(best_res['num_leaves'])
    }
    booster = GradientBoostingClassifier(**params)
    booster.fit(X, y)
    with open(conf.params['train']['model_path'], "wb") as pkl:
        pickle.dump(booster, pkl)

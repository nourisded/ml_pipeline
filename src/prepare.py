import pandas as pd
from sklearn.model_selection import train_test_split
from config_reader import Config

conf = Config()

if __name__ == "__main__":
    train_df = pd.read_csv(conf.params['split']['full_data_path'])
    train_data, val_data = train_test_split(
        train_df, 
        test_size=0.2, 
        random_state = 42, 
        stratify = train_df["price_range"]
    )
    train_data.to_csv(conf.params['split']['train_data_path'], index = False)
    val_data.to_csv(conf.params['split']['eval_data_path'], index = False)
import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["preprocess"]

    raw = np.load("data/raw/train_data.npz")
    x, y = raw["x"] / 255.0, raw["y"]
    
    test_raw = np.load("data/raw/test_data.npz")
    x_test, y_test = test_raw["x"] / 255.0, test_raw["y"]

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=params["val_size"], random_state=params["seed"]
    )

    os.makedirs("data/processed", exist_ok=True)
    np.savez_compressed("data/processed/train.npz", x=x_train, y=y_train)
    np.savez_compressed("data/processed/val.npz", x=x_val, y=y_val)
    np.savez_compressed("data/processed/test.npz", x=x_test, y=y_test)
    print("Processed data saved to data/processed/")

if __name__ == "__main__":
    preprocess()

import os
import yaml
import pandas as pd
import numpy as np
import tensorflow as tf

def train():
    with open("params.yaml") as f:
        p = yaml.safe_load(f)["train"]

    train_data = np.load("data/processed/train.npz")
    val_data = np.load("data/processed/val.npz")

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(p["dense_units"], activation="relu"),
        tf.keras.layers.Dropout(p["dropout_rate"]),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=p["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        train_data["x"], train_data["y"],
        validation_data=(val_data["x"], val_data["y"]),
        epochs=p["epochs"],
        batch_size=p["batch_size"]
    )

    os.makedirs("models", exist_ok=True)
    model.save("models/model.h5")
    pd.DataFrame(history.history).to_csv("models/history.csv", index=False)
    print("Model saved to models/model.h5")

if __name__ == "__main__":
    train()

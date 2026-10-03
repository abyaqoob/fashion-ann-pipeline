import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def evaluate():
    test_data = np.load("data/processed/test.npz")
    model = tf.keras.models.load_model("models/model.h5")

    loss, acc = model.evaluate(test_data["x"], test_data["y"], verbose=0)
    preds = np.argmax(model.predict(test_data["x"]), axis=1)

    cm = confusion_matrix(test_data["y"], preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("models/confusion_matrix.png")

    with open("metrics.json", "w") as f:
        json.dump({"loss": float(loss), "accuracy": float(acc)}, f, indent=4)
    print(f"Test Accuracy: {acc:.4f}, Loss: {loss:.4f}")

if __name__ == "__main__":
    evaluate()

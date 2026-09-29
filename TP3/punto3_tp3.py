'''Implementacion del perceptron multicapa'''
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import ConfusionMatrixDisplay
from data.digit_dataset_loader import load_dataset , get_image , plot_sample
from data.mlpperceptron import MLP, train_mlp, compute_accuracy_mlp, compute_metrics_mlp, MLP2

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = DATA_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

#df = load_dataset(DATA_DIR / "digits.csv")
#df_more = load_dataset(DATA_DIR / "more_digits.csv"

train = pd.concat([load_dataset(DATA_DIR / "digits.csv"), load_dataset(DATA_DIR / "more_digits.csv")])
df = train[~train["image"].apply(lambda a: a.tobytes()).duplicated()].reset_index(drop=True)
print(df.info())

# EDA
print(df.head())
print(df.info())
print(df["label"].value_counts().reindex(range(10), fill_value=0))


X_train = np.stack(df["image"].values)
y_train = np.array(df["label"])

df_test = load_dataset(DATA_DIR / "digits_test.csv")
X_test = np.stack(df_test["image"].values)
y_test = np.array(df_test["label"])

#Transform
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
X_train = X_train.astype(float)
X_test = X_test.astype(float)

#One-hot encoding
encoder = OneHotEncoder(categories=[np.arange(10)], sparse_output=False)
y_train_reshaped = y_train.reshape(-1, 1)
y_test_reshaped = y_test.reshape(-1, 1)
y_train_ohe = encoder.fit_transform(y_train_reshaped)
y_test_ohe = encoder.transform(y_test_reshaped)


#Train
epochs_all = 50
input_features = X_train.shape[1] 
num_classes = y_train_ohe.shape[1]

learning_rates = [0.1, 0.01, 0.001]
architectures = [(32, 16), (64, 32), (128, 64)]
optimizers = ["sgd", "momentum", "rmsprop", "adam"]


#Best Model run:
#Train mlp_2
model_mlp_best = MLP2(
    input_features=X_train.shape[1], 
    hidden1_size= 128,   # Neuronas en la primera capa oculta
    hidden2_size= 64,   # Neuronas en la segunda capa oculta
    output_size=num_classes,
    lr= 0.001,
    acfunc="relu",
    optimizer='momentum'
)
hist3 = train_mlp(model_mlp_best, X_train, y_train_ohe, epochs=epochs_all)
pd.DataFrame(hist3).to_csv(RESULTS_DIR / "pt3hist_mlp_best.csv", index=False)
np.savez(RESULTS_DIR / "pt3mlp_best.npz",
         w1=model_mlp_best.w1, b1=model_mlp_best.b1,
         w2=model_mlp_best.w2, b2=model_mlp_best.b2,
         w3=model_mlp_best.w3, b3=model_mlp_best.b3,
         acfunc=model_mlp_best.acfunc, epochs=epochs_all)

plt.figure(figsize=(12, 6))
plt.plot(hist3["epoch"], hist3["loss"], marker='o', label='MLP_Best')
plt.title('Evolución de la Pérdida por Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.savefig(RESULTS_DIR / "pt3loss_mlp_best.png", bbox_inches="tight")
plt.show() 

train_acc = compute_accuracy_mlp(model_mlp_best,x_train=X_train,y_train=y_train_ohe)
print("Model Accuracy:", train_acc)
precision, recall, f1 = compute_metrics_mlp(model_mlp_best, X_train, y_train_ohe)
print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
test_acc = compute_accuracy_mlp(model_mlp_best, X_test, y_test_ohe)
precision_test, recall_test, f1_test = compute_metrics_mlp(model_mlp_best, X_test, y_test_ohe)
print(f"Accuracy: {test_acc:.4f}")
print(f"Precision: {precision_test:.4f}, Recall: {recall_test:.4f}, F1-Score: {f1_test:.4f}")

y_pred_test = model_mlp_best.forward(X_test).argmax(axis=1)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test, labels=range(10), normalize="true",
                                        values_format=".2f", cmap="Blues")
plt.title("Matriz de confusión en test (normalizada por clase real)")
plt.savefig(RESULTS_DIR / "pt3confusion_matrix_test.png", bbox_inches="tight")
plt.show()


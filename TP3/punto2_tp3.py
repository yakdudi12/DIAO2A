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

df = load_dataset(DATA_DIR / "digits.csv")
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
model_mlp = MLP(input_features=input_features, hidden_size=32, output_size=num_classes, lr=0.01)
hist = train_mlp(model_mlp, X_train, y_train_ohe, epochs=epochs_all)

train_acc = compute_accuracy_mlp(model_mlp,x_train=X_train,y_train=y_train_ohe)
print("Model Accuracy:", train_acc)

precision, recall, f1 = compute_metrics_mlp(model_mlp, X_train, y_train_ohe)
print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

#Train mlp_2
model_mlp2 = MLP2(
    input_features=X_train.shape[1], 
    hidden1_size=64,   # Neuronas en la primera capa oculta
    hidden2_size=32,   # Neuronas en la segunda capa oculta
    output_size=num_classes, 
    lr=0.01,
    acfunc="relu"
)
hist2 = train_mlp(model_mlp2, X_train, y_train_ohe, epochs=epochs_all)

train_acc = compute_accuracy_mlp(model_mlp2,x_train=X_train,y_train=y_train_ohe)
print("Model Accuracy:", train_acc)
precision, recall, f1 = compute_metrics_mlp(model_mlp2, X_train, y_train_ohe)
print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

pd.DataFrame(hist).to_csv(DATA_DIR / "hist_mlp.csv", index=False)
pd.DataFrame(hist2).to_csv(DATA_DIR / "hist_mlp2.csv", index=False)

plt.figure(figsize=(12, 6))
plt.plot(hist["epoch"], hist["loss"], marker='o', label='MLP')
plt.plot(hist2["epoch"], hist2["loss"], marker='*', label='MLP-2')
plt.title('Evolución de la Pérdida por Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show() 



#Grid Search (Opus 5.5)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train_ohe, test_size=0.2, stratify=y_train, random_state=42
)

epochs_gridsearch = 15
learning_rates = [0.1, 0.01, 0.001]
architectures = [(32, 16), (64, 32), (128, 64)]
optimizers = ["sgd", "momentum", "rmsprop", "adam"]
best_acc = 0.0
best_params = {}
all_histories = {}
grid_hists = []

print("Iniciando búsqueda de hiperparámetros...")
for h1, h2 in architectures:
    for lr in learning_rates:
        for opt in optimizers:
            print(f"\n--- Probando Arquitectura: ({h1}, {h2}) | LR: {lr} | Optimizador: {opt} ---")

            # Instanciamos el modelo con las variables del bucle
            model_mlp2 = MLP2(
                input_features=X_train.shape[1],
                hidden1_size=h1,
                hidden2_size=h2,
                output_size=num_classes,
                lr=lr,
                acfunc="relu",
                optimizer=opt
            )

            hist2 = train_mlp(model_mlp2, X_tr, y_tr, epochs=epochs_gridsearch)
            etiqueta = f"Arch({h1},{h2}) - LR:{lr} - {opt}"
            all_histories[etiqueta] = hist2

            # Evaluamos
            train_acc = compute_accuracy_mlp(model_mlp2, x_train=X_tr, y_train=y_tr)
            val_acc = compute_accuracy_mlp(model_mlp2, x_train=X_val, y_train=y_val)
            precision, recall, f1 = compute_metrics_mlp(model_mlp2, X_val, y_val)

            print(f"Accuracy train: {train_acc:.4f} | Accuracy val: {val_acc:.4f} | F1-Score val: {f1:.4f}")
            grid_hists.append(pd.DataFrame(hist2).assign(hidden1=h1, hidden2=h2, lr=lr, optimizer=opt,
                                                         train_acc=train_acc, val_acc=val_acc))

            # Lógica para guardar la mejor combinación
            if val_acc > best_acc:
                best_acc = val_acc
                best_params = {'hidden1': h1, 'hidden2': h2, 'lr': lr, 'optimizer': opt}

print("\n==================================================")
print(f"Búsqueda finalizada. Mejor Accuracy en validación: {best_acc:.4f}")
print(f"Mejores hiperparámetros: {best_params}")
print("==================================================")

pd.concat(grid_hists).to_csv(DATA_DIR / "hist_gridsearch.csv", index=False)

plt.figure(figsize=(14, 8))
for etiqueta, hist_data in all_histories.items():
    plt.plot(hist_data["epoch"], hist_data["loss"], label=etiqueta)
plt.title('Evolución de la Pérdida por Epoch (Grid Search)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
plt.tight_layout()
plt.show()

#Best Model run:
#Train mlp_2
model_mlp_best = MLP2(
    input_features=X_train.shape[1], 
    hidden1_size=best_params['hidden1'],   # Neuronas en la primera capa oculta
    hidden2_size=best_params['hidden2'],   # Neuronas en la segunda capa oculta
    output_size=num_classes,
    lr=best_params['lr'],
    acfunc="relu",
    optimizer=best_params['optimizer']
)
hist3 = train_mlp(model_mlp_best, X_train, y_train_ohe, epochs=epochs_all)
pd.DataFrame(hist3).to_csv(DATA_DIR / "hist_mlp_best.csv", index=False)
np.savez(DATA_DIR / "mlp_best.npz",
         w1=model_mlp_best.w1, b1=model_mlp_best.b1,
         w2=model_mlp_best.w2, b2=model_mlp_best.b2,
         w3=model_mlp_best.w3, b3=model_mlp_best.b3,
         acfunc=model_mlp_best.acfunc, epochs=epochs_all, **best_params)

plt.figure(figsize=(12, 6))
plt.plot(hist3["epoch"], hist3["loss"], marker='o', label='MLP_Best')
plt.title('Evolución de la Pérdida por Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
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
plt.show()


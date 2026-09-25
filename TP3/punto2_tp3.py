'''Implementacion del perceptron multicapa'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from data.digit_dataset_loader import load_dataset , get_image , plot_sample
from data.mlpperceptron import MLP, train_mlp, compute_accuracy_mlp, compute_metrics_mlp, MLP2

df = load_dataset(fr"TP3/data/digits.csv")
print(df.info())

X_train = np.stack(df["image"].values)
y_train = np.array(df["label"])

df_test = load_dataset(fr"TP3/data/digits_test.csv")
X_test = np.stack(df_test["image"].values)
y_test = np.array(df_test["label"])

#Transform
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
X_train = X_train.astype(float) / 255.0 
X_test = X_test.astype(float) / 255.0

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

plt.figure(figsize=(12, 6))
plt.plot(hist["epoch"], hist["loss"], marker='o', label='MLP')
plt.plot(hist2["epoch"], hist2["loss"], marker='*', label='MLP-2')
plt.title('Evolución de la Pérdida por Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show() 



#Grid Search
epochs_gridsearch = 15
learning_rates = [0.1, 0.01, 0.001]
architectures = [(32, 16), (64, 32), (128, 64)] 
best_acc = 0.0
best_params = {}
all_histories = {}

print("Iniciando búsqueda de hiperparámetros...")
for h1, h2 in architectures:
    for lr in learning_rates:
        print(f"\n--- Probando Arquitectura: ({h1}, {h2}) | LR: {lr} ---")
        
        # Instanciamos el modelo con las variables del bucle
        model_mlp2 = MLP2(
            input_features=X_train.shape[1], 
            hidden1_size=h1,   
            hidden2_size=h2,   
            output_size=num_classes, 
            lr=lr,
            acfunc="relu"
        )
        
        hist2 = train_mlp(model_mlp2, X_train, y_train_ohe, epochs=epochs_gridsearch)
        etiqueta = f"Arch({h1},{h2}) - LR:{lr}"
        all_histories[etiqueta] = hist2

        # Evaluamos
        train_acc = compute_accuracy_mlp(model_mlp2, x_train=X_train, y_train=y_train_ohe)
        precision, recall, f1 = compute_metrics_mlp(model_mlp2, X_train, y_train_ohe)
        
        print(f"Accuracy: {train_acc:.4f} | F1-Score: {f1:.4f}")
        
        # Lógica para guardar la mejor combinación
        if train_acc > best_acc:
            best_acc = train_acc
            best_params = {'hidden1': h1, 'hidden2': h2, 'lr': lr}

print("\n==================================================")
print(f"Búsqueda finalizada. Mejor Accuracy: {best_acc:.4f}")
print(f"Mejores hiperparámetros: {best_params}")
print("==================================================")

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
    hidden1_size=128,   # Neuronas en la primera capa oculta
    hidden2_size=64,   # Neuronas en la segunda capa oculta
    output_size=num_classes, 
    lr=0.01,
    acfunc="relu"
)
hist3 = train_mlp(model_mlp_best, X_train, y_train_ohe, epochs=epochs_all)

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


'''Implementacion del perceptron simple y multicapa Santiago'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from data.perceptron import PerceptronLineal , PerceptronNoLineal , train , compute_accuracy, compute_metrics
from data_profiling import ProfileReport

path_df = fr"TP3\data\fraud_dataset.csv"
df = pd.read_csv(path_df,encoding='UTF-8')
df = df.drop('big_model_fraud_probability', axis=1)

# EDA
print(df.head())
print(df.info())
#profile = ProfileReport(df, title="EDA Fraud Dataset", explorative=True)
#profile.to_file(fr"TP3\reporte_eda.html")


# Perceptrones
features_cols = ["account_age_days", "amount_usd", "days_since_last_purchase", "items_viewed_before_purchase",
                 "quantity_purchased", "session_duration_seconds" ]

X = df[features_cols].values
y = df['flagged_fraud'].values

#Normalizar features:
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
X_train = X_scaled
y_train = y

#Entrenar Perceptrones
print("\nPerceptron Lineal\n")
ppn = PerceptronLineal(num_features=X_train.shape[1], lr=0.01)
hist_lineal = train(ppn, X_train, y_train, epochs = 10)
train_acc = compute_accuracy(ppn,x_train=X_train,y_train=y_train)
print("Model Accuracy:", train_acc)

precision, recall, f1 = compute_metrics(ppn, X_train, y_train)
print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

print("\nPerceptron No lineal (BCE + Sigmoid)\n")
ppn_nolineal = PerceptronNoLineal(num_features=X_train.shape[1], lr=0.01)
hist_nolineal = train(ppn_nolineal, X_train, y_train, epochs = 10)
train_acc_2 = compute_accuracy(ppn_nolineal,x_train=X_train,y_train=y_train)
print("Model Accuracy:", train_acc_2)

precision2, recall2, f1_2 = compute_metrics(ppn_nolineal, X_train, y_train)
print(f"Precision: {precision2}, Recall: {recall2}, F1-Score: {f1_2}")


plt.figure(figsize=(12, 6))
plt.plot(hist_nolineal["epoch"], hist_nolineal["loss"], marker='o', label='Perceptrón No Lineal')
plt.plot(hist_lineal["epoch"], hist_lineal["loss"], marker='*', label='Perceptrón Lineal')
plt.title('Evolución de la Pérdida por Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()

# Prueba de generalizacion del modelo Perceptron No Lineal:
from sklearn.preprocessing import StandardScaler
#Train test split y Standar
X_train_2, X_test, y_train_2, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_2)
X_test_scaled = scaler.transform(X_test)

print("\nPerceptron No lineal (BCE + Sigmoid)\n")
ppn_nolineal_2 = PerceptronNoLineal(num_features=X_train_scaled.shape[1], lr=0.01)
hist_nolineal_2 = train(ppn_nolineal_2, X_train_scaled, y_train_2, epochs = 10)

#Train
train_acc_2 = compute_accuracy(ppn_nolineal_2,x_train=X_train_scaled,y_train=y_train_2)
print("Model Accuracy en Train:", train_acc_2)
precision_train, recall_train, f1_train = compute_metrics(ppn_nolineal_2, X_train_scaled, y_train_2)
print(f"Precision: {precision_train}, Recall: {recall_train}, F1-Score: {f1_train}")

#Test
acc_nonlin = compute_accuracy(ppn_nolineal_2, X_test_scaled, y_test)
print("\nModel Accuracy en Test:", train_acc_2)

prec_nonlin, rec_nonlin, f1_nonlin = compute_metrics(ppn_nolineal_2, X_test_scaled, y_test)
print("\nModel Metrics en Test:")
print(f"Precision: {prec_nonlin}, Recall: {rec_nonlin}, F1-Score: {f1_nonlin}")



'''plt.figure(figsize=(12, 6))
plt.plot(hist_nolineal["epoch"], hist_nolineal["loss"], marker='o', label='Perceptrón No Lineal')
plt.plot(hist_lineal["epoch"], hist_lineal["loss"], marker='*', label='Perceptrón Lineal')
plt.title('Evolución de la Pérdida por Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()'''

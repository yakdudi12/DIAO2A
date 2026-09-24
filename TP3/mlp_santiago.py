'''Implementacion del perceptron simple y multicapa Santiago'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from data.digit_dataset_loader import load_dataset , get_image , plot_sample
from data.perceptron import PerceptronLineal , PerceptronNoLineal , train , compute_accuracy, plot_boundary, compute_metrics

df_raw = load_dataset(fr"TP3\data\digits.csv")
print(df_raw.info())
df = df_raw[df_raw["label"].isin([0, 1])]

X = np.stack(df["image"].values)
y = np.array(df["label"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
print(X_train.shape)


ppn = PerceptronLineal(num_features=X_train.shape[1], lr=0.01)
train(ppn, X_train, y_train, epochs = 10)
train_acc = compute_accuracy(ppn,x_train=X_train,y_train=y_train)
print("Model Accuracy:", train_acc)

precision, recall, f1 = compute_metrics(ppn, X_train, y_train)
print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

ppn_nolineal = PerceptronNoLineal(num_features=X_train.shape[1], lr=0.01)
train(ppn_nolineal, X_train, y_train, epochs = 10)
train_acc_2 = compute_accuracy(ppn_nolineal,x_train=X_train,y_train=y_train)
print("Model Accuracy:", train_acc_2)

precision2, recall2, f1_2 = compute_metrics(ppn_nolineal, X_train, y_train)
print(f"Precision: {precision2}, Recall: {recall2}, F1-Score: {f1_2}")

'''Implementacion del perceptron simple y multicapa Santiago'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from data.digit_dataset_loader import load_dataset , get_image , plot_sample
from data.perceptron import Perceptron , train , compute_accuracy, plot_boundary, compute_metrics
import tqdm

df_raw = load_dataset(fr"/home/santiago/PrincialSanty/ITBA_Cursos/SIA/DIA02A/TP3/data/digits_test.csv")
print(df_raw.info())
df = df_raw[df_raw["label"].isin([0, 1])]

X = np.stack(df["image"].values)
y = np.array(df["label"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
print(X_train.shape)


ppn = Perceptron(num_features=X_train.shape[1])
train(ppn, X_train, y_train, epochs = 5)
train_acc = compute_accuracy(ppn,x_train=X_train,y_train=y_train)
print("Model Accuracy:", train_acc)

precision, recall, f1 = compute_metrics(ppn, X_train, y_train)
print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

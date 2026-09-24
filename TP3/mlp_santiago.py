'''Implementacion del perceptron simple y multicapa Santiago'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from data.digit_dataset_loader import load_dataset , get_image , plot_sample

df = load_dataset(fr"/home/santiago/PrincialSanty/ITBA_Cursos/SIA/DIA02A/TP3/data/digits_test.csv")
plot_sample(df.iloc[0])

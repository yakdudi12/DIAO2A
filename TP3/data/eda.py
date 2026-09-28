import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_profiling import ProfileReport

path_df = fr"TP3\data\fraud_dataset.csv"
df = pd.read_csv(path_df,encoding='UTF-8')

df = df.drop('big_model_fraud_probability', axis=1)

print(df.head())
print(df.info())

profile = ProfileReport(df, title="EDA Fraud Dataset", explorative=True)
profile.to_file(fr"TP3\reporte_eda.html")
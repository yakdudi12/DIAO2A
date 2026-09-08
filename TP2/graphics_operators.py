import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

# Cargar los datos
path = r"TP2/resultados_mejorados/comparacion_operadores.csv"
outpath = r"TP2/resultados_mejorados/"
df = pd.read_csv(path)

df['configuracion'] = df.apply(
    lambda row: f"{row['selection']} + {row['crossover']} + {row['mutation']} + {row['survival']}", 
    axis=1
)

df_sorted = df.sort_values('error').reset_index(drop=True)

# GRÁFICO 1: Lineplot general de las 96 corridas
plt.figure(figsize=(12, 5))
sns.lineplot(x=df_sorted.index, y=df_sorted['error'], marker="o", color="tab:blue")
plt.title("Error final de las 96 configuraciones (ordenadas de mejor a peor)", fontsize=14, weight='bold')
plt.xlabel("Ranking de Configuración (0 = La Mejor, 95 = La Peor)")
plt.ylabel("Error")
plt.fill_between(df_sorted.index, df_sorted['error'], color="tab:blue", alpha=0.1)
plt.tight_layout()
plt.savefig(fr"{outpath}/error_configs.png")
plt.show()

# GRÁFICO 2: Barplot con el TOP 15 con MENOR error
plt.figure(figsize=(10, 8))
top_15 = df_sorted.head(15)
ax = sns.barplot(data=top_15, x='error', y='configuracion', palette="viridis")
plt.title("Top 15: Configuraciones con MENOR Error", fontsize=14, weight='bold')
plt.xlabel("Error de Aproximación")
plt.ylabel("Configuración (Selección + Cruza + Mutación + Supervivencia)")

# Agregar el valor del error al lado de cada barra para más claridad
for p in ax.patches:
    ax.annotate(f"{p.get_width():.5f}", 
                (p.get_width(), p.get_y() + p.get_height() / 2.), 
                ha='left', va='center', xytext=(5, 0), textcoords='offset points')
plt.xlim(top_15['error'].min() * 0.95, top_15['error'].max() * 1.05) # Ajustar zoom al rango del top 15
plt.tight_layout()
plt.savefig(fr"{outpath}/barplot_errors.png")
plt.show()


# GRÁFICO 3: Impacto individual de cada operador (Boxplots)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Distribución del Error por Operador (Menor es mejor)", fontsize=16, weight='bold')

sns.boxplot(data=df, x='selection', y='error', ax=axes[0, 0], palette="Set2")
axes[0, 0].set_title("Efecto de la Selección")

sns.boxplot(data=df, x='crossover', y='error', ax=axes[0, 1], palette="Set2")
axes[0, 1].set_title("Efecto de la Cruza")

sns.boxplot(data=df, x='mutation', y='error', ax=axes[1, 0], palette="Set2")
axes[1, 0].set_title("Efecto de la Mutación")

sns.boxplot(data=df, x='survival', y='error', ax=axes[1, 1], palette="Set2")
axes[1, 1].set_title("Efecto de la Supervivencia")

for ax in axes.flat:
    ax.tick_params(axis='x', rotation=15)
plt.tight_layout()
plt.savefig(fr"{outpath}/boxplot_errors.png")
plt.show()


# GRÁFICO 4: Trade-off de Tiempo vs Error (Scatterplot)
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df, 
    x='seconds', 
    y='error', 
    hue='mutation',    
    style='survival', 
    s=100, 
    alpha=0.8,
    palette="deep"
)
plt.title("Compromiso: Tiempo de Ejecución vs Error Final", fontsize=14, weight='bold')
plt.xlabel("Tiempo (segundos)")
plt.ylabel("Error")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(fr"{outpath}/scatter_errorandtime.png")
plt.show()
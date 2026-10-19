"""Selección de creadores por campaña.

Supuestos de negocio (el enunciado permite asumirlos):
- Los clientes venden o reciben adopciones en Argentina, así que la audiencia útil es el
  alcance estimado en Argentina (`estimated_argentina_reach`).
- La fracción de esa audiencia interesada en el rubro es proporcional al porcentaje de
  publicaciones del creador sobre esa temática.
- Guardados, comentarios y visitas al perfil son interacciones de **intención**: indican
  que la persona quiere volver al contenido, preguntar o conocer más. Likes y compartidos
  son reacciones rápidas y no se cuentan.
"""

import pandas as pd

from preprocessing import TOPIC_LABELS

TOPIC_COLUMN = {label: column for column, label in TOPIC_LABELS.items()}

# Métrica que minimiza cada tipo de objetivo.
OBJECTIVES = {
    "intención": "USD por interacción de intención",
    "alcance": "USD por 1000 de alcance relevante",
}

# Los cuatro clientes buscan acciones (adopciones o ventas), no sólo visibilidad.
CLIENTS = {
    "Refugio de perros": {"topic": "Perros", "objective": "intención"},
    "Ropa de fútbol": {"topic": "Fútbol", "objective": "intención"},
    "Productos de belleza": {"topic": "Maquillaje", "objective": "intención"},
    "Materiales de arte": {"topic": "Arte", "objective": "intención"},
}


def campaign_metrics(df: pd.DataFrame, topic: str) -> pd.DataFrame:
    """Métricas por creador para una publicación patrocinada en una campaña del rubro `topic`."""
    share = df[TOPIC_COLUMN[topic]] / 100
    intent_rate = (df["saves_per_1000"] + df["comments_per_1000"] + df["profile_visits_per_1000"]) / 1000
    relevant_reach = df["estimated_argentina_reach"] * share
    relevant_intent = relevant_reach * intent_rate
    return pd.DataFrame({
        "price_usd": df["price_usd"],
        "alcance relevante": relevant_reach,
        "intención relevante": relevant_intent,
        OBJECTIVES["alcance"]: 1000 * df["price_usd"] / relevant_reach,
        OBJECTIVES["intención"]: df["price_usd"] / relevant_intent,
    })


def select(df: pd.DataFrame, topic: pd.Series, client_topic: str, objective: str, n: int = 3) -> pd.Index:
    """Los `n` creadores de la temática del cliente con menor costo según el objetivo."""
    metric = campaign_metrics(df, client_topic)[OBJECTIVES[objective]]
    return metric[topic == client_topic].nsmallest(n).index


def compare_strategies(df: pd.DataFrame, topic: pd.Series, index: pd.Series, n: int = 3) -> pd.DataFrame:
    """Para cada cliente, la selección propuesta frente a dos selecciones por seguidores."""
    rows = []
    for client, spec in CLIENTS.items():
        metrics = campaign_metrics(df, spec["topic"])
        strategies = {
            "Top followers (actual)": df["followers"].nlargest(n).index,
            "Top followers de la temática": df.loc[topic == spec["topic"], "followers"].nlargest(n).index,
            "Temática + costo por intención": select(df, topic, spec["topic"], spec["objective"], n),
        }
        for strategy, chosen in strategies.items():
            selected = metrics.loc[chosen]
            cost = selected["price_usd"].sum()
            rows.append({
                "cliente": client,
                "estrategia": strategy,
                "creadores": ", ".join(df.loc[chosen, "name"]),
                "de la temática": int((topic.loc[chosen] == spec["topic"]).sum()),
                "índice Oja medio": index.loc[chosen].mean(),
                "costo USD": cost,
                "alcance relevante": selected["alcance relevante"].sum(),
                "intención relevante": selected["intención relevante"].sum(),
                OBJECTIVES["alcance"]: 1000 * cost / selected["alcance relevante"].sum(),
                OBJECTIVES["intención"]: cost / selected["intención relevante"].sum(),
            })
    return pd.DataFrame(rows).set_index(["cliente", "estrategia"])

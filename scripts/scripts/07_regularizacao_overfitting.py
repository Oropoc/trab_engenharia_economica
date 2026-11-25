"""Etapa 6: comparação entre MLP sem e com regularização e análise de curvas."""

from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# %% CAMINHOS DO PROJETO

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
if not (RAIZ_PROJETO / "data").exists():
    raiz_alternativa = RAIZ_PROJETO.parent
    if (raiz_alternativa / "data").exists():
        RAIZ_PROJETO = raiz_alternativa

ARQUIVO_LIMPO = (
    RAIZ_PROJETO
    / "data"
    / "processed"
    / "ProgramaRodofaunaAtropelada_silvestre_limpo.xlsx"
)
PASTA_RESULTADOS = RAIZ_PROJETO / "resultados"
PASTA_FIGURAS = PASTA_RESULTADOS / "figuras"
PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)

if not ARQUIVO_LIMPO.exists():
    raise FileNotFoundError(
        "Dataset limpo não encontrado. Execute primeiro scripts/01_preparacao_dados.py."
    )


# %% DADOS E PRÉ-PROCESSAMENTO (MESMO PADRÃO DO SCRIPT 03)

df = pd.read_excel(ARQUIVO_LIMPO)
df["data_dt"] = pd.to_datetime(df["data_dt"], errors="coerce")
if df["data_dt"].isna().any():
    raise ValueError("O dataset contém datas inválidas.")

df["ano_data"] = df["data_dt"].dt.year
df["mes_data"] = df["data_dt"].dt.month
df["dia_semana_data"] = df["data_dt"].dt.day_of_week

colunas_numericas = ["ano_data", "x", "y"]
colunas_categoricas = ["mes_data", "dia_semana_data", "uc"]
X = df[colunas_numericas + colunas_categoricas].copy()
y = df["classe"].copy()

X_treino, X_temp, y_treino, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_teste, y_val, y_teste = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

prep = ColumnTransformer(
    [
        ("num", StandardScaler(), colunas_numericas),
        ("cat", ohe, colunas_categoricas),
    ]
)

X_treino_p = prep.fit_transform(X_treino)
X_val_p = prep.transform(X_val)
X_teste_p = prep.transform(X_teste)
classes = np.sort(y_treino.unique())


# %% TREINAMENTO POR ÉPOCA PARA REGISTRAR CURVAS DE TREINO E VALIDAÇÃO

def treinar_com_curvas(
    nome: str,
    hidden_layer_sizes: tuple[int, ...],
    alpha: float,
    usar_early_stopping: bool,
    epocas: int = 120,
    paciencia: int = 20,
):
    modelo = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=alpha,
        batch_size=64,
        learning_rate_init=0.001,
        random_state=42,
    )

    hist_loss_treino = []
    hist_loss_val = []
    hist_f1_treino = []
    hist_f1_val = []

    melhor_loss_val = float("inf")
    melhor_modelo = None
    melhor_epoca = 0
    sem_melhora = 0

    for epoca in range(1, epocas + 1):
        modelo.partial_fit(X_treino_p, y_treino, classes=classes)

        prob_treino = modelo.predict_proba(X_treino_p)
        prob_val = modelo.predict_proba(X_val_p)
        pred_treino = classes[np.argmax(prob_treino, axis=1)]
        pred_val = classes[np.argmax(prob_val, axis=1)]

        loss_t = log_loss(y_treino, prob_treino, labels=classes)
        loss_v = log_loss(y_val, prob_val, labels=classes)
        f1_t = f1_score(y_treino, pred_treino, average="macro", zero_division=0)
        f1_v = f1_score(y_val, pred_val, average="macro", zero_division=0)

        hist_loss_treino.append(loss_t)
        hist_loss_val.append(loss_v)
        hist_f1_treino.append(f1_t)
        hist_f1_val.append(f1_v)

        if loss_v < melhor_loss_val - 1e-4:
            melhor_loss_val = loss_v
            melhor_modelo = deepcopy(modelo)
            melhor_epoca = epoca
            sem_melhora = 0
        else:
            sem_melhora += 1

        if usar_early_stopping and sem_melhora >= paciencia:
            break

    modelo_final = melhor_modelo if (usar_early_stopping and melhor_modelo is not None) else modelo
    pred_teste = modelo_final.predict(X_teste_p)

    resumo = {
        "configuracao": nome,
        "epocas_executadas": len(hist_loss_treino),
        "melhor_epoca_val": melhor_epoca,
        "loss_treino_final": hist_loss_treino[-1],
        "loss_val_final": hist_loss_val[-1],
        "f1_macro_treino_final": hist_f1_treino[-1],
        "f1_macro_val_final": hist_f1_val[-1],
        "acuracia_teste": accuracy_score(y_teste, pred_teste),
        "acuracia_balanceada_teste": balanced_accuracy_score(y_teste, pred_teste),
        "f1_macro_teste": f1_score(y_teste, pred_teste, average="macro", zero_division=0),
    }

    curvas = {
        "loss_treino": hist_loss_treino,
        "loss_val": hist_loss_val,
        "f1_treino": hist_f1_treino,
        "f1_val": hist_f1_val,
    }
    return resumo, curvas


configuracoes = [
    ("Sem regularização (128, 64), alpha=0", (128, 64), 0.0, False),
    ("Com L2 (128, 64), alpha=0.05", (128, 64), 0.05, False),
    ("L2 + Early Stopping (128, 64), alpha=0.05", (128, 64), 0.05, True),
    ("Complexidade reduzida + L2 + Early Stopping (32,)", (32,), 0.01, True),
]

resultados = []
curvas_por_modelo = {}

for nome, arq, alpha, es in configuracoes:
    resumo, curvas = treinar_com_curvas(nome, arq, alpha, es)
    resultados.append(resumo)
    curvas_por_modelo[nome] = curvas

df_res = pd.DataFrame(resultados)
arquivo_csv = PASTA_RESULTADOS / "regularizacao_mlp.csv"
df_res.to_csv(arquivo_csv, index=False, encoding="utf-8-sig")

print("\nCOMPARAÇÃO DE REGULARIZAÇÃO NA MLP")
print("-" * 70)
print(df_res.round(4).to_string(index=False))


# %% GRÁFICO DE CURVAS DE LOSS (TREINO VS. VALIDAÇÃO)

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=True)
for ax, (nome, curvas) in zip(axes.ravel(), curvas_por_modelo.items()):
    ep = range(1, len(curvas["loss_treino"]) + 1)
    ax.plot(ep, curvas["loss_treino"], label="Loss treino", color="#1d3557")
    ax.plot(ep, curvas["loss_val"], label="Loss validação", color="#e63946", linestyle="--")
    ax.set_title(nome, fontsize=10)
    ax.set_xlabel("Época")
    ax.set_ylabel("Log-loss")
    ax.legend()

fig.suptitle("Curvas de Loss (Treino × Validação) — Análise de Overfitting", fontsize=12)
fig.tight_layout()
arquivo_fig = PASTA_FIGURAS / "07_curvas_regularizacao_mlp.png"
fig.savefig(arquivo_fig, dpi=150)
plt.close(fig)

print(f"\nTabela salva em: {arquivo_csv}")
print(f"Curvas salvas em: {arquivo_fig}")
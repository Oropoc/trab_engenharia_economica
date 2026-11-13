"""Prevê contagens anuais registradas por célula espacial de 1 km."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RAIZ_PROJETO = Path(__file__).resolve().parents[1]
ARQUIVO_LIMPO = (
    RAIZ_PROJETO
    / "data"
    / "processed"
    / "ProgramaRodofaunaAtropelada_silvestre_limpo.xlsx"
)
PASTA_RESULTADOS = RAIZ_PROJETO / "resultados"
PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
TAMANHO_CELULA_METROS = 1000

if not ARQUIVO_LIMPO.exists():
    raise FileNotFoundError(
        "Dataset limpo não encontrado. Execute primeiro "
        "scripts/01_preparacao_dados.py."
    )

df = pd.read_excel(ARQUIVO_LIMPO)
df["data_dt"] = pd.to_datetime(df["data_dt"], errors="coerce")
if df["data_dt"].isna().any():
    raise ValueError("O dataset contém datas inválidas.")
if df[["x", "y"]].isna().any().any():
    raise ValueError("O dataset contém coordenadas ausentes.")

df["ano"] = df["data_dt"].dt.year.astype(int)
df["cell_x"] = np.floor(df["x"] / TAMANHO_CELULA_METROS).astype(int)
df["cell_y"] = np.floor(df["y"] / TAMANHO_CELULA_METROS).astype(int)

contagens = (
    df.groupby(["cell_x", "cell_y", "ano"])
    .size()
    .rename("ocorrencias")
    .reset_index()
)
anos = sorted(df["ano"].unique().tolist())
anos_treino = anos[:-2]
celulas = (
    df.loc[df["ano"].isin(anos_treino), ["cell_x", "cell_y"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
celulas_total = df[["cell_x", "cell_y"]].drop_duplicates()
celulas_novas_futuras = len(celulas_total) - len(celulas)

# Zeros significam que não houve registro naquela célula/ano; não comprovam
# ausência de atropelamentos, pois a cobertura de busca não está no dataset.
painel = celulas.assign(_chave=1).merge(
    pd.DataFrame({"ano": anos, "_chave": 1}),
    on="_chave",
).drop(columns="_chave")
painel = painel.merge(
    contagens,
    on=["cell_x", "cell_y", "ano"],
    how="left",
)
painel["ocorrencias"] = painel["ocorrencias"].fillna(0).astype(int)
painel["x_centro_m"] = (
    painel["cell_x"] + 0.5
) * TAMANHO_CELULA_METROS
painel["y_centro_m"] = (
    painel["cell_y"] + 0.5
) * TAMANHO_CELULA_METROS

if len(anos) < 4:
    raise ValueError(
        "A regressão temporal exige ao menos quatro anos de dados: "
        "anos iniciais para treino, um para validação e um para teste."
    )

ano_teste = anos[-1]
ano_validacao = anos[-2]
if len(anos_treino) < 2:
    raise ValueError("Não há anos suficientes para treino e validação cruzada.")

colunas_entrada = ["x_centro_m", "y_centro_m", "ano"]
X = painel[colunas_entrada]
y = painel["ocorrencias"]

mask_treino = painel["ano"].isin(anos_treino)
mask_validacao = painel["ano"].eq(ano_validacao)
mask_teste = painel["ano"].eq(ano_teste)
X_treino, y_treino = X.loc[mask_treino], y.loc[mask_treino]
X_validacao, y_validacao = X.loc[mask_validacao], y.loc[mask_validacao]
X_teste, y_teste = X.loc[mask_teste], y.loc[mask_teste]

if X_treino.empty or X_validacao.empty or X_teste.empty:
    raise ValueError("Um dos períodos de treino, validação ou teste está vazio.")

cv = GroupKFold(n_splits=len(anos_treino))
grupos_treino = painel.loc[mask_treino, "ano"]

pipeline_rf = Pipeline(
    [
        ("escala", StandardScaler()),
        (
            "modelo",
            RandomForestRegressor(
                random_state=42,
                n_jobs=1,
            ),
        ),
    ]
)
busca_rf = GridSearchCV(
    pipeline_rf,
    {
        "modelo__n_estimators": [150],
        "modelo__max_depth": [None, 12],
        "modelo__min_samples_leaf": [2, 5],
    },
    scoring="neg_mean_absolute_error",
    cv=cv,
    n_jobs=1,
    refit=True,
    error_score="raise",
)
busca_rf.fit(X_treino, y_treino, groups=grupos_treino)

pipeline_mlp = Pipeline(
    [
        ("escala", StandardScaler()),
        (
            "modelo",
            MLPRegressor(
                max_iter=1000,
                early_stopping=True,
                n_iter_no_change=25,
                random_state=42,
            ),
        ),
    ]
)
busca_mlp = GridSearchCV(
    pipeline_mlp,
    {
        "modelo__hidden_layer_sizes": [(32,), (64, 32)],
        "modelo__alpha": [0.001, 0.01],
    },
    scoring="neg_mean_absolute_error",
    cv=cv,
    n_jobs=1,
    refit=True,
    error_score="raise",
)
busca_mlp.fit(X_treino, y_treino, groups=grupos_treino)

modelos = {
    "Dummy (média)": (
        DummyRegressor(strategy="mean").fit(X_treino, y_treino),
        None,
    ),
    "Poisson": (
        Pipeline(
            [
                ("escala", StandardScaler()),
                ("modelo", PoissonRegressor(max_iter=1000)),
            ]
        ).fit(X_treino, y_treino),
        None,
    ),
    "Random Forest otimizada": (
        busca_rf.best_estimator_,
        float(-busca_rf.best_score_),
    ),
    "MLP otimizada": (
        busca_mlp.best_estimator_,
        float(-busca_mlp.best_score_),
    ),
}


def calcular_metricas(observado, previsto):
    return {
        "MAE": mean_absolute_error(observado, previsto),
        "RMSE": np.sqrt(mean_squared_error(observado, previsto)),
        "R2": r2_score(observado, previsto),
    }


resultados_validacao = []
for nome, (modelo, mae_cv) in modelos.items():
    previsao = np.maximum(modelo.predict(X_validacao), 0)
    metricas = calcular_metricas(y_validacao, previsao)
    resultados_validacao.append(
        {
            "modelo": nome,
            "MAE_validacao": metricas["MAE"],
            "RMSE_validacao": metricas["RMSE"],
            "R2_validacao": metricas["R2"],
            "MAE_CV_treino": mae_cv,
        }
    )

comparacao = pd.DataFrame(resultados_validacao).sort_values(
    "MAE_validacao"
)
nome_escolhido = str(comparacao.iloc[0]["modelo"])
modelo_escolhido = modelos[nome_escolhido][0]

mask_treino_validacao = painel["ano"].le(ano_validacao)
modelo_final = clone(modelo_escolhido).fit(
    X.loc[mask_treino_validacao],
    y.loc[mask_treino_validacao],
)
previsoes_teste = np.maximum(modelo_final.predict(X_teste), 0)
metricas_teste = calcular_metricas(y_teste, previsoes_teste)

arquivo_comparacao = PASTA_RESULTADOS / "regressao_contagem_modelos.csv"
comparacao.to_csv(arquivo_comparacao, index=False, encoding="utf-8-sig")

arquivo_previsoes = PASTA_RESULTADOS / "previsoes_contagem_teste.csv"
pd.DataFrame(
    {
        "cell_x": painel.loc[mask_teste, "cell_x"].to_numpy(),
        "cell_y": painel.loc[mask_teste, "cell_y"].to_numpy(),
        "x_centro_m": X_teste["x_centro_m"].to_numpy(),
        "y_centro_m": X_teste["y_centro_m"].to_numpy(),
        "ano": ano_teste,
        "ocorrencias_observadas": y_teste.to_numpy(),
        "ocorrencias_previstas": previsoes_teste,
        "modelo": nome_escolhido,
    }
).to_csv(arquivo_previsoes, index=False, encoding="utf-8-sig")

print(
    f"Células espaciais observadas: {len(celulas)}; "
    f"tamanho da célula: {TAMANHO_CELULA_METROS} m."
)
print(
    "Células que só aparecem após o período de treino e ficaram fora da "
    f"previsão: {celulas_novas_futuras}."
)
print(
    f"Treino: {anos_treino}; validação: {ano_validacao}; "
    f"teste temporal reservado: {ano_teste}."
)
print("\nMétricas no ano de validação:")
print(comparacao.round(4).to_string(index=False))
print(f"\nModelo selecionado pela validação: {nome_escolhido}")
print(f"Métricas no teste temporal ({ano_teste}): {metricas_teste}")
print(f"Comparação salva em: {arquivo_comparacao}")
print(f"Previsões do teste salvas em: {arquivo_previsoes}")
print(
    "\nInterpretação: a variável prevista é a contagem de registros por "
    "célula/ano, apenas para células registradas no período de treino. "
    "Sem dados de esforço de busca, ela não representa probabilidade ou "
    "risco absoluto de atropelamento."
)

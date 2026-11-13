"""Compara e otimiza modelos para prever a classe taxonômica."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression


RAIZ_PROJETO = Path(__file__).resolve().parents[1]
ARQUIVO_LIMPO = (
    RAIZ_PROJETO
    / "data"
    / "processed"
    / "ProgramaRodofaunaAtropelada_silvestre_limpo.xlsx"
)
PASTA_RESULTADOS = RAIZ_PROJETO / "resultados"
PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

if not ARQUIVO_LIMPO.exists():
    raise FileNotFoundError(
        "Dataset limpo não encontrado. Execute primeiro "
        "scripts/01_preparacao_dados.py."
    )

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

if y.nunique() < 2:
    raise ValueError("A classificação exige pelo menos duas classes.")

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

preprocessador = ColumnTransformer(
    transformers=[
        ("numericas", StandardScaler(), colunas_numericas),
        (
            "categoricas",
            OneHotEncoder(handle_unknown="ignore"),
            colunas_categoricas,
        ),
    ],
    sparse_threshold=0,
)

validacao_cruzada = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42,
)

modelos = {
    "Dummy (classe mais frequente)": (
        DummyClassifier(strategy="most_frequent"),
        {},
    ),
    "Regressão logística": (
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        ),
        {"modelo__C": [0.1, 1.0, 10.0]},
    ),
    "Random Forest": (
        RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        ),
        {
            "modelo__n_estimators": [150],
            "modelo__max_depth": [None, 15],
            "modelo__min_samples_leaf": [2, 5],
        },
    ),
    "MLP": (
        MLPClassifier(
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=25,
            random_state=42,
        ),
        {
            "modelo__hidden_layer_sizes": [(32,), (64, 32)],
            "modelo__alpha": [0.001, 0.01],
        },
    ),
}

resultados = []
relatorios = []

print(f"Treino: {len(X_treino)} registros; teste reservado: {len(X_teste)}.")
print("Ajuste de hiperparâmetros e validação cruzada acontecem apenas no treino.")

for nome, (estimador, grade) in modelos.items():
    pipeline = Pipeline(
        [
            ("preprocessador", preprocessador),
            ("modelo", estimador),
        ]
    )

    if grade:
        busca = GridSearchCV(
            pipeline,
            param_grid=grade,
            scoring="f1_macro",
            cv=validacao_cruzada,
            n_jobs=1,
            refit=True,
            error_score="raise",
        )
        busca.fit(X_treino, y_treino)
        modelo_final = busca.best_estimator_
        f1_cv = float(busca.best_score_)
        melhores_parametros = str(busca.best_params_)
    else:
        f1_cv = float(
            cross_val_score(
                pipeline,
                X_treino,
                y_treino,
                scoring="f1_macro",
                cv=validacao_cruzada,
                n_jobs=1,
            ).mean()
        )
        modelo_final = pipeline.fit(X_treino, y_treino)
        melhores_parametros = "sem ajuste"

    previsoes_teste = modelo_final.predict(X_teste)
    f1_teste = f1_score(
        y_teste,
        previsoes_teste,
        average="macro",
        zero_division=0,
    )
    resultados.append(
        {
            "modelo": nome,
            "f1_macro_cv_treino": f1_cv,
            "f1_macro_teste": f1_teste,
            "diferenca_cv_teste": f1_cv - f1_teste,
            "acuracia_teste": accuracy_score(y_teste, previsoes_teste),
            "acuracia_balanceada_teste": balanced_accuracy_score(
                y_teste,
                previsoes_teste,
            ),
            "melhores_parametros": melhores_parametros,
        }
    )
    relatorios.append(
        f"=== {nome} ===\n"
        + classification_report(
            y_teste,
            previsoes_teste,
            zero_division=0,
        )
    )
    print(
        f"{nome}: F1-macro CV={f1_cv:.4f}; "
        f"F1-macro teste={f1_teste:.4f}"
    )

arquivo_comparacao = PASTA_RESULTADOS / "classificacao_ajustada.csv"
pd.DataFrame(resultados).sort_values(
    "f1_macro_cv_treino",
    ascending=False,
).to_csv(arquivo_comparacao, index=False, encoding="utf-8-sig")

arquivo_relatorios = PASTA_RESULTADOS / "relatorios_classificacao_teste.txt"
arquivo_relatorios.write_text(
    "\n\n".join(relatorios),
    encoding="utf-8",
)

print(f"\nComparação salva em: {arquivo_comparacao}")
print(f"Relatórios por classe salvos em: {arquivo_relatorios}")

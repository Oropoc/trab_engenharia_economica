"""Classificação multiclasse da classe do animal atropelado.

Modelos implementados neste arquivo:
- baseline Dummy, que sempre prevê a classe mais frequente;
- baseline de regressão logística balanceada.

A primeira MLP será adicionada na Missão 12.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# %% CAMINHOS E LEITURA

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
PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

if not ARQUIVO_LIMPO.exists():
    raise FileNotFoundError(
        "Dataset limpo não encontrado. Execute primeiro:\n"
        "scripts/01_preparacao_dados.py"
    )

df = pd.read_excel(ARQUIVO_LIMPO)
df["data_dt"] = pd.to_datetime(df["data_dt"], errors="coerce")

if df["data_dt"].isna().any():
    raise ValueError("O arquivo limpo contém datas inválidas.")


# %% 1. DEFINIÇÃO DAS ENTRADAS E DO ALVO

df["ano_data"] = df["data_dt"].dt.year
df["mes_data"] = df["data_dt"].dt.month
df["dia_semana_data"] = df["data_dt"].dt.dayofweek

colunas_entrada = [
    "ano_data",
    "mes_data",
    "dia_semana_data",
    "uc",
    "x",
    "y",
]
coluna_alvo = "classe"

X = df[colunas_entrada].copy()
y = df[coluna_alvo].copy()

assert len(X) == len(y)
assert X.isna().sum().sum() == 0
assert y.isna().sum() == 0

print("Formato de X:", X.shape)
print("Formato de y:", y.shape)


# %% 2. SEPARAÇÃO EM TREINO, VALIDAÇÃO E TESTE

X_treino, X_temporario, y_treino, y_temporario = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

X_validacao, X_teste, y_validacao, y_teste = train_test_split(
    X_temporario,
    y_temporario,
    test_size=0.50,
    random_state=42,
    stratify=y_temporario,
)

del X_temporario
del y_temporario

assert set(X_treino.index).isdisjoint(X_validacao.index)
assert set(X_treino.index).isdisjoint(X_teste.index)
assert set(X_validacao.index).isdisjoint(X_teste.index)
assert len(X_treino) + len(X_validacao) + len(X_teste) == len(X)

print("\nTAMANHO DOS CONJUNTOS")
print("-" * 50)
print(f"Treino: {len(X_treino)}")
print(f"Validação: {len(X_validacao)}")
print(f"Teste reservado: {len(X_teste)}")


def percentual_por_classe(serie: pd.Series) -> pd.Series:
    return (
        serie
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .sort_index()
    )


distribuicao_classes = pd.concat(
    {
        "Completo": percentual_por_classe(y),
        "Treino": percentual_por_classe(y_treino),
        "Validação": percentual_por_classe(y_validacao),
        "Teste": percentual_por_classe(y_teste),
    },
    axis=1,
)

print("\nPERCENTUAL DE CADA CLASSE")
print("-" * 50)
print(distribuicao_classes.to_string())


# %% 3. PRÉ-PROCESSAMENTO APRENDIDO SOMENTE NO TREINO

colunas_numericas = ["ano_data", "x", "y"]
colunas_categoricas = ["mes_data", "dia_semana_data", "uc"]

try:
    codificador_categorico = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )
except TypeError:
    # Compatibilidade com versões antigas do scikit-learn.
    codificador_categorico = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False,
    )

preprocessador = ColumnTransformer(
    transformers=[
        (
            "numericas",
            StandardScaler(),
            colunas_numericas,
        ),
        (
            "categoricas",
            codificador_categorico,
            colunas_categoricas,
        ),
    ]
)

X_treino_preparado_array = preprocessador.fit_transform(X_treino)
X_validacao_preparado_array = preprocessador.transform(X_validacao)
X_teste_preparado_array = preprocessador.transform(X_teste)

nomes_colunas_preparadas = preprocessador.get_feature_names_out()

X_treino_preparado = pd.DataFrame(
    X_treino_preparado_array,
    columns=nomes_colunas_preparadas,
    index=X_treino.index,
)
X_validacao_preparado = pd.DataFrame(
    X_validacao_preparado_array,
    columns=nomes_colunas_preparadas,
    index=X_validacao.index,
)
X_teste_preparado = pd.DataFrame(
    X_teste_preparado_array,
    columns=nomes_colunas_preparadas,
    index=X_teste.index,
)

assert not X_treino_preparado.isna().any().any()
assert not X_validacao_preparado.isna().any().any()
assert not X_teste_preparado.isna().any().any()

print("\nFORMATOS APÓS O PRÉ-PROCESSAMENTO")
print("-" * 50)
print("Treino:", X_treino_preparado.shape)
print("Validação:", X_validacao_preparado.shape)
print("Teste reservado:", X_teste_preparado.shape)


# %% 4. FUNÇÃO ÚNICA DE AVALIAÇÃO NA VALIDAÇÃO

def avaliar_na_validacao(nome_modelo, modelo, previsoes) -> dict:
    """Exibe métricas comparáveis sem consultar o conjunto de teste."""

    metricas = {
        "Acurácia": accuracy_score(y_validacao, previsoes),
        "Acurácia balanceada": balanced_accuracy_score(
            y_validacao,
            previsoes,
        ),
        "F1-macro": f1_score(
            y_validacao,
            previsoes,
            average="macro",
            zero_division=0,
        ),
    }

    print(f"\nRESULTADOS: {nome_modelo}")
    print("-" * 50)
    for metrica, valor in metricas.items():
        print(f"{metrica}: {valor:.4f}")

    print("\nRelatório por classe:")
    print(
        classification_report(
            y_validacao,
            previsoes,
            zero_division=0,
        )
    )

    classes = modelo.classes_
    matriz = pd.DataFrame(
        confusion_matrix(
            y_validacao,
            previsoes,
            labels=classes,
        ),
        index=[f"Real: {classe}" for classe in classes],
        columns=[f"Previsto: {classe}" for classe in classes],
    )

    print("Matriz de confusão:")
    print(matriz.to_string())

    return metricas


# %% 5. BASELINE DUMMY

modelo_dummy = DummyClassifier(strategy="most_frequent")
modelo_dummy.fit(X_treino_preparado, y_treino)

previsoes_dummy = modelo_dummy.predict(X_validacao_preparado)
metricas_dummy = avaliar_na_validacao(
    "Dummy — classe mais frequente",
    modelo_dummy,
    previsoes_dummy,
)


# %% 6. BASELINE DE REGRESSÃO LOGÍSTICA BALANCEADA

modelo_logistico = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    random_state=42,
)
modelo_logistico.fit(X_treino_preparado, y_treino)

previsoes_logistico = modelo_logistico.predict(X_validacao_preparado)
metricas_logistico = avaliar_na_validacao(
    "Regressão logística balanceada",
    modelo_logistico,
    previsoes_logistico,
)


# %% 7. COMPARAÇÃO DOS BASELINES

comparacao_modelos = pd.DataFrame(
    {
        "Dummy": metricas_dummy,
        "Regressão logística balanceada": metricas_logistico,
    }
).T

print("\nCOMPARAÇÃO DOS MODELOS")
print("-" * 50)
print(comparacao_modelos.round(4).to_string())

arquivo_resultados = PASTA_RESULTADOS / "baselines_multiclasse.csv"
comparacao_modelos.to_csv(arquivo_resultados, encoding="utf-8-sig")
print(f"\nResultados salvos em: {arquivo_resultados}")


# %% MISSÃO 12 - PRIMEIRA REDE NEURAL MLP

from sklearn.neural_network import MLPClassifier



modelo_mlp_inicial = MLPClassifier(
    hidden_layer_sizes=(32,),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    batch_size=64,
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=25,
    random_state=42
)


modelo_mlp_inicial.fit(
    X_treino_preparado,
    y_treino
)


print("\nINFORMAÇÕES DO TREINAMENTO")
print("--------------------------------")

print(
    "Épocas executadas:",
    modelo_mlp_inicial.n_iter_
)

print(
    "Valor final da função de perda:",
    round(modelo_mlp_inicial.loss_, 4)
)

print(
    "Melhor acurácia na validação interna:",
    round(
        max(modelo_mlp_inicial.validation_scores_),
        4
    )
)



previsoes_mlp = modelo_mlp_inicial.predict(
    X_validacao_preparado
)

acuracia_mlp = accuracy_score(
    y_validacao,
    previsoes_mlp
)

acuracia_balanceada_mlp = balanced_accuracy_score(
    y_validacao,
    previsoes_mlp
)

f1_macro_mlp = f1_score(
    y_validacao,
    previsoes_mlp,
    average="macro",
    zero_division=0
)

print("\nRESULTADOS DA MLP INICIAL")
print("--------------------------------")

print(f"Acurácia: {acuracia_mlp:.4f}")
print(
    "Acurácia balanceada: "
    f"{acuracia_balanceada_mlp:.4f}"
)
print(f"F1-macro: {f1_macro_mlp:.4f}")



comparacao_modelos = pd.DataFrame(
    {
        "Dummy": {
            "Acurácia": acuracia_baseline,
            "Acurácia balanceada": acuracia_balanceada_baseline,
            "F1-macro": f1_macro_baseline
        },
        "Regressão logística balanceada": {
            "Acurácia": acuracia_logistico,
            "Acurácia balanceada": acuracia_balanceada_logistico,
            "F1-macro": f1_macro_logistico
        },
        "MLP inicial": {
            "Acurácia": acuracia_mlp,
            "Acurácia balanceada": acuracia_balanceada_mlp,
            "F1-macro": f1_macro_mlp
        }
    }
).T

print("\nCOMPARAÇÃO DOS MODELOS")
print("--------------------------------")
print(comparacao_modelos.round(4))


print("\nRELATÓRIO DA MLP INICIAL")
print("--------------------------------")

print(
    classification_report(
        y_validacao,
        previsoes_mlp,
        zero_division=0
    )
)
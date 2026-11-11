"""Análise exploratória do dataset silvestre já limpo.

Este script descreve os dados que efetivamente serão usados na modelagem.
Ele não altera o arquivo limpo e não treina modelos.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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
PASTA_FIGURAS = RAIZ_PROJETO / "resultados" / "figuras"
PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)

if not ARQUIVO_LIMPO.exists():
    raise FileNotFoundError(
        "Dataset limpo não encontrado. Execute primeiro:\n"
        "scripts/01_preparacao_dados.py"
    )


# %% 1. LEITURA

df = pd.read_excel(ARQUIVO_LIMPO)
df["data_dt"] = pd.to_datetime(df["data_dt"], errors="coerce")

print("DATASET SILVESTRE LIMPO")
print("-" * 50)
print(f"Linhas: {df.shape[0]}")
print(f"Colunas: {df.shape[1]}")
print(f"IDs distintos: {df['id'].nunique(dropna=False)}")
print(f"Data inicial: {df['data_dt'].min().date()}")
print(f"Data final: {df['data_dt'].max().date()}")


# %% 2. COMPLETUDE E CARDINALIDADE

resumo_colunas = pd.DataFrame(
    {
        "tipo": df.dtypes.astype(str),
        "nulos": df.isna().sum(),
        "percentual_nulos": (df.isna().mean() * 100).round(2),
        "valores_distintos": df.nunique(dropna=False),
    }
).sort_values("percentual_nulos", ascending=False)

print("\nRESUMO DAS COLUNAS")
print("-" * 50)
print(resumo_colunas.to_string())

# Valores como "nao_identificado" não são NaN, mas também representam ausência.
colunas_textuais = df.select_dtypes(include=["object", "string"]).columns
sentinelas_por_coluna = {}

for coluna in colunas_textuais:
    texto = df[coluna].astype("string").str.strip().str.lower()
    quantidade = texto.str.contains(
        r"nao.identific|não.identific",
        regex=True,
        na=False,
    ).sum()
    if quantidade > 0:
        sentinelas_por_coluna[coluna] = int(quantidade)

print("\nVALORES SENTINELA 'NÃO IDENTIFICADO'")
print("-" * 50)
print(sentinelas_por_coluna)


# %% 3. DISTRIBUIÇÃO DAS CLASSES

quantidade_classes = df["classe"].value_counts()
percentual_classes = (
    df["classe"].value_counts(normalize=True).mul(100).round(2)
)

resumo_classes = pd.DataFrame(
    {
        "quantidade": quantidade_classes,
        "percentual": percentual_classes,
    }
)

print("\nDISTRIBUIÇÃO DAS CLASSES")
print("-" * 50)
print(resumo_classes.to_string())

fig, ax = plt.subplots(figsize=(8, 5))
quantidade_classes.sort_values().plot.barh(ax=ax, color="#2b7a78")
ax.set_title("Registros silvestres por classe")
ax.set_xlabel("Quantidade de registros")
ax.set_ylabel("Classe")
fig.tight_layout()
fig.savefig(PASTA_FIGURAS / "01_distribuicao_classes.png", dpi=150)
plt.close(fig)


# %% 4. DISTRIBUIÇÃO TEMPORAL

df["ano_data"] = df["data_dt"].dt.year
df["mes_data"] = df["data_dt"].dt.month

registros_por_ano = df["ano_data"].value_counts().sort_index()
registros_por_mes = df["mes_data"].value_counts().sort_index()

print("\nREGISTROS POR ANO")
print("-" * 50)
print(registros_por_ano.to_string())

print("\nREGISTROS POR MÊS")
print("-" * 50)
print(registros_por_mes.to_string())

fig, ax = plt.subplots(figsize=(8, 5))
registros_por_ano.plot.bar(ax=ax, color="#3aafa9")
ax.set_title("Registros silvestres por ano")
ax.set_xlabel("Ano")
ax.set_ylabel("Quantidade de registros")
ax.tick_params(axis="x", rotation=0)
fig.tight_layout()
fig.savefig(PASTA_FIGURAS / "02_registros_por_ano.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
registros_por_mes.reindex(range(1, 13), fill_value=0).plot.bar(
    ax=ax,
    color="#def2f1",
    edgecolor="#17252a",
)
ax.set_title("Registros silvestres por mês")
ax.set_xlabel("Mês")
ax.set_ylabel("Quantidade de registros")
ax.tick_params(axis="x", rotation=0)
fig.tight_layout()
fig.savefig(PASTA_FIGURAS / "03_registros_por_mes.png", dpi=150)
plt.close(fig)


# %% 5. DISTRIBUIÇÃO ESPACIAL E UNIDADES DE CONSERVAÇÃO

print("\nUNIDADES DE CONSERVAÇÃO")
print("-" * 50)
print(df["uc"].value_counts(dropna=False).to_string())

print("\nRESUMO DAS COORDENADAS")
print("-" * 50)
print(df[["x", "y"]].describe().to_string())

fig, ax = plt.subplots(figsize=(8, 7))

for classe, grupo in df.groupby("classe"):
    ax.scatter(
        grupo["x"],
        grupo["y"],
        s=12,
        alpha=0.45,
        label=classe,
    )

ax.set_title("Distribuição espacial dos registros por classe")
ax.set_xlabel("Coordenada X")
ax.set_ylabel("Coordenada Y")
ax.legend(title="Classe")
fig.tight_layout()
fig.savefig(PASTA_FIGURAS / "04_distribuicao_espacial.png", dpi=150)
plt.close(fig)


# %% 6. POSSÍVEIS VALORES EXTREMOS

def contar_extremos_iqr(serie: pd.Series) -> tuple[int, float, float]:
    """Conta observações fora de 1,5 vezes o intervalo interquartil."""

    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    quantidade = (
        (serie < limite_inferior) | (serie > limite_superior)
    ).sum()
    return int(quantidade), float(limite_inferior), float(limite_superior)


print("\nPOSSÍVEIS EXTREMOS PELO CRITÉRIO DO IQR")
print("-" * 50)

for coluna in ["x", "y"]:
    quantidade, limite_inferior, limite_superior = contar_extremos_iqr(
        df[coluna]
    )
    print(
        f"{coluna}: {quantidade} registros fora de "
        f"[{limite_inferior:.2f}, {limite_superior:.2f}]"
    )

print(
    "\nObservação: coordenadas extremas podem ser locais geográficos válidos; "
    "não devem ser removidas automaticamente."
)
print(f"\nFiguras salvas em: {PASTA_FIGURAS}")

"""Preparação e limpeza do dataset de atropelamentos de fauna.

Este script:
1. lê e audita o arquivo original;
2. comprova a duplicação dos IDs;
3. remove somente as duplicatas comprovadamente idênticas;
4. filtra a fauna silvestre;
5. trata a classe sem identificação e a UC ausente;
6. salva versões intermediárias e a versão limpa.

Ele não realiza análise exploratória nem treina modelos.
"""

from pathlib import Path

import pandas as pd


# %% CAMINHOS DO PROJETO

RAIZ_PROJETO = Path(__file__).resolve().parents[1]

# Aceita também o caso em que a pasta "scripts" esteja dentro de uma
# subpasta do projeto, como "trab_engenharia_economica/scripts".
if not (RAIZ_PROJETO / "data").exists():
    raiz_alternativa = RAIZ_PROJETO.parent
    if (raiz_alternativa / "data").exists():
        RAIZ_PROJETO = raiz_alternativa
PASTA_RAW = RAIZ_PROJETO / "data" / "raw"
PASTA_PROCESSADOS = RAIZ_PROJETO / "data" / "processed"

ARQUIVO_ORIGINAL = PASTA_RAW / "ProgramaRodofaunaAtropelada_original.xlsx"
ARQUIVO_SEM_DUPLICATAS = (
    PASTA_PROCESSADOS / "ProgramaRodofaunaAtropelada_sem_duplicatas.xlsx"
)
ARQUIVO_SILVESTRE = (
    PASTA_PROCESSADOS / "ProgramaRodofaunaAtropelada_silvestre.xlsx"
)
ARQUIVO_LIMPO = (
    PASTA_PROCESSADOS / "ProgramaRodofaunaAtropelada_silvestre_limpo.xlsx"
)

PASTA_PROCESSADOS.mkdir(parents=True, exist_ok=True)

if not ARQUIVO_ORIGINAL.exists():
    raise FileNotFoundError(
        "Arquivo original não encontrado. Coloque-o em:\n"
        f"{ARQUIVO_ORIGINAL}"
    )


# %% 1. LEITURA E AUDITORIA INICIAL DO ARQUIVO BRUTO

print("Arquivo que será lido:")
print(ARQUIVO_ORIGINAL)

df_bruto = pd.read_excel(ARQUIVO_ORIGINAL)

print("\nDataset carregado com sucesso.")
print(df_bruto.head())

print("\nRESUMO DO DATASET ORIGINAL")
print("-" * 40)
print(f"Linhas: {df_bruto.shape[0]}")
print(f"Colunas: {df_bruto.shape[1]}")
print(f"IDs distintos: {df_bruto['id'].nunique(dropna=False)}")

print("\nRegistros por tipo de fauna:")
print(df_bruto["silvestre_"].value_counts(dropna=False))

print("\nTipos das colunas:")
print(df_bruto.dtypes)

print("\nValores nulos no arquivo bruto:")
print(df_bruto.isna().sum().sort_values(ascending=False))


# %% 2. COMPROVAÇÃO DAS DUPLICATAS

quantidade_por_id = df_bruto.groupby("id", dropna=False).size()

print("\nAUDITORIA DOS IDs")
print("-" * 40)
print(quantidade_por_id.value_counts().sort_index())

if not (quantidade_por_id == 2).all():
    raise ValueError(
        "Nem todos os IDs aparecem exatamente duas vezes. "
        "A regra de remoção precisa ser revista antes de continuar."
    )

# OBJECTID identifica a linha e, por isso, não entra na comparação do conteúdo.
colunas_comparacao = [
    coluna
    for coluna in df_bruto.columns
    if coluna not in ["OBJECTID", "id"]
]

quantidade_valores_por_id = (
    df_bruto
    .groupby("id", dropna=False)[colunas_comparacao]
    .nunique(dropna=False)
)

ids_com_conteudo_diferente = quantidade_valores_por_id.index[
    quantidade_valores_por_id.gt(1).any(axis=1)
]

print(
    "IDs cujas duas linhas possuem alguma diferença de conteúdo:",
    len(ids_com_conteudo_diferente),
)

if len(ids_com_conteudo_diferente) != 0:
    raise ValueError(
        "Foram encontrados IDs com conteúdos diferentes. "
        "A remoção automática foi interrompida."
    )

print("Duplicação comprovada: as duas linhas de cada ID são equivalentes.")


# %% 3. REMOÇÃO DAS DUPLICATAS COMPROVADAS

df_sem_duplicatas = (
    df_bruto
    .sort_values(["id", "OBJECTID"])
    .drop_duplicates(subset="id", keep="first")
    .copy()
    .reset_index(drop=True)
)

assert df_sem_duplicatas["id"].is_unique
assert len(df_sem_duplicatas) == df_bruto["id"].nunique(dropna=False)

df_sem_duplicatas.to_excel(ARQUIVO_SEM_DUPLICATAS, index=False)

print("\nDUPLICATAS REMOVIDAS")
print("-" * 40)
print(f"Linhas antes: {len(df_bruto)}")
print(f"Linhas depois: {len(df_sem_duplicatas)}")
print(f"Arquivo salvo em: {ARQUIVO_SEM_DUPLICATAS}")


# %% 4. FILTRO PARA FAUNA SILVESTRE

tipo_fauna_normalizado = (
    df_sem_duplicatas["silvestre_"]
    .astype("string")
    .str.strip()
    .str.lower()
)

df_silvestre = (
    df_sem_duplicatas
    .loc[tipo_fauna_normalizado == "silvestre"]
    .copy()
    .reset_index(drop=True)
)

assert df_silvestre["id"].is_unique
assert (
    df_silvestre["silvestre_"]
    .astype("string")
    .str.strip()
    .str.lower()
    .eq("silvestre")
    .all()
)

df_silvestre.to_excel(ARQUIVO_SILVESTRE, index=False)

print("\nFILTRO DA FAUNA SILVESTRE")
print("-" * 40)
print(f"Registros silvestres: {len(df_silvestre)}")
print(f"Registros removidos pelo filtro: {len(df_sem_duplicatas) - len(df_silvestre)}")
print(f"Arquivo salvo em: {ARQUIVO_SILVESTRE}")


# %% 5. TRATAMENTO DOS VALORES NECESSÁRIOS À MODELAGEM

df_limpo = df_silvestre.copy()

classe_normalizada = (
    df_limpo["classe"]
    .astype("string")
    .str.strip()
    .str.lower()
)

classe_invalida = (
    classe_normalizada.isna()
    | classe_normalizada.str.contains("nao_identific", na=False)
    | classe_normalizada.str.contains("não identific", na=False)
)

print("\nREGISTROS SEM CLASSE VÁLIDA")
print("-" * 40)
print(df_limpo.loc[classe_invalida, ["id", "classe"]])
print(f"Quantidade: {classe_invalida.sum()}")

df_limpo = (
    df_limpo
    .loc[~classe_invalida]
    .copy()
    .reset_index(drop=True)
)

df_limpo["uc"] = (
    df_limpo["uc"]
    .astype("string")
    .str.strip()
    .replace("", pd.NA)
    .fillna("nao_identificada")
)

df_limpo["data_dt"] = pd.to_datetime(
    df_limpo["data_dt"],
    errors="coerce",
)

if df_limpo["data_dt"].isna().any():
    raise ValueError("Existem datas inválidas após a conversão de data_dt.")

assert df_limpo["id"].is_unique
assert df_limpo["classe"].isna().sum() == 0
assert df_limpo["uc"].isna().sum() == 0

df_limpo.to_excel(ARQUIVO_LIMPO, index=False)

print("\nPREPARAÇÃO CONCLUÍDA")
print("-" * 40)
print(f"Registros finais: {len(df_limpo)}")
print(f"Arquivo limpo salvo em: {ARQUIVO_LIMPO}")

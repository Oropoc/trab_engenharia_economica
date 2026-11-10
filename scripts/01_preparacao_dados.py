# %% ---------------------------------------------------------
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# -------------------------------------------------------------

from pathlib import Path

import pandas as pd



# %% ---------------------------------------------------------
# 2. LOCALIZAÇÃO DO PROJETO E DO DATASET
# -------------------------------------------------------------

# __file__ representa o arquivo Python atual.
# parent representa a pasta "scripts".
# parent.parent representa a pasta principal do projeto.
pasta_projeto = Path(__file__).resolve().parent.parent

caminho_arquivo = (
    pasta_projeto
    / "data"
    / "raw"
    / "ProgramaRodofaunaAtropelada_original.xlsx"
)

print("Arquivo que será lido:")
print(caminho_arquivo)



# %% ---------------------------------------------------------
# 3. LEITURA DO DATASET ORIGINAL
# -------------------------------------------------------------

df_bruto = pd.read_excel(caminho_arquivo)

print("Dataset carregado com sucesso.")



# %% ---------------------------------------------------------
# 4. VISUALIZAÇÃO DAS PRIMEIRAS LINHAS
# -------------------------------------------------------------

print(df_bruto.head())



# %% ---------------------------------------------------------
# 5. TAMANHO DO DATASET
# -------------------------------------------------------------

quantidade_linhas = df_bruto.shape[0]
quantidade_colunas = df_bruto.shape[1]

print("Quantidade de linhas:", quantidade_linhas)
print("Quantidade de colunas:", quantidade_colunas)



# %% ---------------------------------------------------------
# 6. VERIFICAÇÃO DOS IDENTIFICADORES
# -------------------------------------------------------------

quantidade_ids_distintos = df_bruto["id"].nunique()

print("Quantidade de linhas:", quantidade_linhas)
print("Quantidade de IDs distintos:", quantidade_ids_distintos)



# %% ---------------------------------------------------------
# 7. CONTAGEM DE FAUNA SILVESTRE E DOMÉSTICA
# -------------------------------------------------------------

print(df_bruto["silvestre_"].value_counts())



# %% ---------------------------------------------------------
# 8. RESUMO INICIAL
# -------------------------------------------------------------

print()
print("RESUMO DO DATASET ORIGINAL")
print("--------------------------")
print(f"Linhas: {quantidade_linhas}")
print(f"Colunas: {quantidade_colunas}")
print(f"IDs distintos: {quantidade_ids_distintos}")
print()
print("Registros por tipo:")
print(df_bruto["silvestre_"].value_counts())



# %% ---------------------------------------------------------
# 9. QUANTIDADE DE OCORRÊNCIAS DE CADA ID
# -------------------------------------------------------------

contagem_por_id = df_bruto.groupby("id").size()

print("Exemplo da contagem dos primeiros IDs:")
print(contagem_por_id.head())

print()
print("Resumo das repetições:")
print(contagem_por_id.value_counts().sort_index())



# %% ---------------------------------------------------------
# 10. VERIFICAÇÃO DA QUANTIDADE DE REPETIÇÕES
# -------------------------------------------------------------

ids_com_quantidade_incorreta = contagem_por_id[
    contagem_por_id != 2
]

print(
    "IDs que não aparecem exatamente duas vezes:",
    len(ids_com_quantidade_incorreta)
)



# %% ---------------------------------------------------------
# 11. COLUNAS QUE SERÃO COMPARADAS
# -------------------------------------------------------------

colunas_ignoradas = ["OBJECTID", "id"]

colunas_comparacao = [
    coluna
    for coluna in df_bruto.columns
    if coluna not in colunas_ignoradas
]

print("Colunas que serão comparadas:")
print(colunas_comparacao)

print()
print("Quantidade de colunas comparadas:", len(colunas_comparacao))



# %% ---------------------------------------------------------
# 12. COMPARAÇÃO DO CONTEÚDO DAS LINHAS
# -------------------------------------------------------------

valores_distintos_por_id = (
    df_bruto
    .groupby("id")[colunas_comparacao]
    .nunique(dropna=False)
)

print(valores_distintos_por_id.head())



# %% ---------------------------------------------------------
# 13. IDENTIFICAÇÃO DE POSSÍVEIS DIFERENÇAS
# -------------------------------------------------------------

possui_diferenca = (
    valores_distintos_por_id > 1
).any(axis=1)

ids_com_diferencas = possui_diferenca[
    possui_diferenca
].index.tolist()

print(
    "Quantidade de IDs com alguma diferença:",
    len(ids_com_diferencas)
)



# %% ---------------------------------------------------------
# 14. RESULTADO DA VERIFICAÇÃO
# -------------------------------------------------------------

if len(ids_com_quantidade_incorreta) == 0 and len(ids_com_diferencas) == 0:
    print("DUPLICAÇÃO COMPROVADA")
    print("Todos os IDs aparecem exatamente duas vezes.")
    print("As duas linhas de cada ID possuem os mesmos dados.")
    print("A única diferença esperada está no OBJECTID.")
else:
    print("ATENÇÃO")
    print("Foram encontradas diferenças que precisam ser analisadas.")

    print("IDs com quantidade incorreta:")
    print(ids_com_quantidade_incorreta.head())

    print("IDs com valores diferentes:")
    print(ids_com_diferencas[:10])
    
    

# %% ---------------------------------------------------------
# 15. EXEMPLO DE UM PAR DUPLICADO
# -------------------------------------------------------------

exemplo_id = 1

exemplo_duplicado = df_bruto[
    df_bruto["id"] == exemplo_id
]

print(exemplo_duplicado.to_string(index=False))



# %% ---------------------------------------------------------
# 16. TESTES AUTOMÁTICOS DA DUPLICAÇÃO
# -------------------------------------------------------------

assert (contagem_por_id == 2).all(), (
    "Existem IDs que não aparecem exatamente duas vezes."
)

assert len(ids_com_diferencas) == 0, (
    "Existem pares com valores diferentes."
)

print("Testes automáticos concluídos com sucesso.")



# %% ---------------------------------------------------------
# 17. CRIAÇÃO DO DATASET SEM DUPLICATAS
# -------------------------------------------------------------

df_sem_duplicatas = (
    df_bruto
    .sort_values(["id", "OBJECTID"])
    .drop_duplicates(subset="id", keep="first")
    .copy()
    .reset_index(drop=True)
)

print("Dataset sem duplicatas criado.")



# %% ---------------------------------------------------------
# 18. COMPARAÇÃO ANTES E DEPOIS
# -------------------------------------------------------------

linhas_antes = len(df_bruto)
linhas_depois = len(df_sem_duplicatas)
linhas_removidas = linhas_antes - linhas_depois

print("Linhas antes:", linhas_antes)
print("Linhas depois:", linhas_depois)
print("Cópias removidas:", linhas_removidas)



# %% ---------------------------------------------------------
# 19. VERIFICAÇÃO DA PRESERVAÇÃO DO DATASET BRUTO
# -------------------------------------------------------------

print("Linhas em df_bruto:", len(df_bruto))
print("Linhas em df_sem_duplicatas:", len(df_sem_duplicatas))



# %% ---------------------------------------------------------
# 20. VERIFICAÇÃO DE DUPLICATAS RESTANTES
# -------------------------------------------------------------

quantidade_ids_repetidos = (
    df_sem_duplicatas["id"]
    .duplicated()
    .sum()
)

print(
    "Quantidade de IDs ainda repetidos:",
    quantidade_ids_repetidos
)



# %% ---------------------------------------------------------
# 21. CONTAGEM APÓS A REMOÇÃO DAS DUPLICATAS
# -------------------------------------------------------------

print(
    df_sem_duplicatas["silvestre_"].value_counts()
)



# %% ---------------------------------------------------------
# 22. TESTES DO DATASET SEM DUPLICATAS
# -------------------------------------------------------------

assert len(df_bruto) == 10328, (
    "O dataset bruto foi alterado."
)

assert len(df_sem_duplicatas) == 5164, (
    "A quantidade de registros após a remoção está incorreta."
)

assert df_sem_duplicatas["id"].nunique() == 5164, (
    "A quantidade de IDs distintos está incorreta."
)

assert not df_sem_duplicatas["id"].duplicated().any(), (
    "Ainda existem IDs duplicados."
)

print("Todos os testes da remoção de duplicatas foram aprovados.")



# %% ---------------------------------------------------------
# 23. SALVAMENTO DA VERSÃO SEM DUPLICATAS
# -------------------------------------------------------------

caminho_sem_duplicatas = (
    pasta_projeto
    / "data"
    / "processed"
    / "ProgramaRodofaunaAtropelada_sem_duplicatas.xlsx"
)

df_sem_duplicatas.to_excel(
    caminho_sem_duplicatas,
    index=False
)

print("Arquivo salvo em:")
print(caminho_sem_duplicatas)



# %% ---------------------------------------------------------
# 24. VERIFICAÇÃO DO ARQUIVO SALVO
# -------------------------------------------------------------

df_verificacao = pd.read_excel(
    caminho_sem_duplicatas
)

assert len(df_verificacao) == 5164
assert df_verificacao["id"].nunique() == 5164
assert not df_verificacao["id"].duplicated().any()

print("Arquivo salvo e verificado com sucesso.")



# %% ---------------------------------------------------------
# 25. VERIFICAÇÃO DAS CATEGORIAS DE FAUNA
# -------------------------------------------------------------

contagem_tipo_fauna = (
    df_sem_duplicatas["silvestre_"]
    .value_counts(dropna=False)
)

print("Categorias encontradas:")
print(contagem_tipo_fauna)



# %% ---------------------------------------------------------
# 26. VALORES DISTINTOS DA COLUNA SILVESTRE_
# -------------------------------------------------------------

categorias_fauna = (
    df_sem_duplicatas["silvestre_"]
    .unique()
)

print("Valores encontrados em silvestre_:")
print(categorias_fauna)



# %% ---------------------------------------------------------
# 27. FILTRO DA FAUNA SILVESTRE
# -------------------------------------------------------------

df_silvestre = (
    df_sem_duplicatas
    .loc[
        df_sem_duplicatas["silvestre_"] == "silvestre"
    ]
    .copy()
    .reset_index(drop=True)
)

print("Dataset de fauna silvestre criado.")



# %% ---------------------------------------------------------
# 28. COMPARAÇÃO DO FILTRO DE ESCOPO
# -------------------------------------------------------------

quantidade_antes_filtro = len(df_sem_duplicatas)
quantidade_depois_filtro = len(df_silvestre)

quantidade_domesticos_removidos = (
    quantidade_antes_filtro
    - quantidade_depois_filtro
)

percentual_domesticos = (
    quantidade_domesticos_removidos
    / quantidade_antes_filtro
    * 100
)

print(
    "Registros antes do filtro:",
    quantidade_antes_filtro
)

print(
    "Registros silvestres:",
    quantidade_depois_filtro
)

print(
    "Registros domésticos retirados:",
    quantidade_domesticos_removidos
)

print(
    f"Percentual retirado: {percentual_domesticos:.2f}%"
)



# %% ---------------------------------------------------------
# 29. VERIFICAÇÃO DO RESULTADO DO FILTRO
# -------------------------------------------------------------

print(
    df_silvestre["silvestre_"]
    .value_counts(dropna=False)
)

quantidade_domesticos_restantes = (
    df_silvestre["silvestre_"]
    .eq("domestico")
    .sum()
)

print(
    "Animais domésticos restantes:",
    quantidade_domesticos_restantes
)



# %% ---------------------------------------------------------
# 30. VERIFICAÇÃO DOS IDENTIFICADORES
# -------------------------------------------------------------

print(
    "Quantidade de registros:",
    len(df_silvestre)
)

print(
    "Quantidade de IDs distintos:",
    df_silvestre["id"].nunique()
)

print(
    "Quantidade de IDs repetidos:",
    df_silvestre["id"].duplicated().sum()
)



# %% ---------------------------------------------------------
# 31. PRESERVAÇÃO DAS VERSÕES ANTERIORES
# -------------------------------------------------------------

print("Linhas em df_bruto:", len(df_bruto))

print(
    "Linhas em df_sem_duplicatas:",
    len(df_sem_duplicatas)
)

print(
    "Linhas em df_silvestre:",
    len(df_silvestre)
)



# %% ---------------------------------------------------------
# 32. TESTES AUTOMÁTICOS DO FILTRO
# -------------------------------------------------------------

assert len(df_bruto) == 10328, (
    "O dataset bruto foi alterado."
)

assert len(df_sem_duplicatas) == 5164, (
    "O dataset sem duplicatas foi alterado."
)

assert len(df_silvestre) == 4422, (
    "A quantidade de registros silvestres está incorreta."
)

assert df_silvestre["id"].nunique() == 4422, (
    "A quantidade de IDs silvestres está incorreta."
)

assert not df_silvestre["id"].duplicated().any(), (
    "Existem IDs repetidos no dataset silvestre."
)

assert (
    df_silvestre["silvestre_"] == "silvestre"
).all(), (
    "O dataset ainda contém registros fora do escopo silvestre."
)

print("Todos os testes do filtro silvestre foram aprovados.")



# %% ---------------------------------------------------------
# 33. SALVAMENTO DO DATASET SILVESTRE
# -------------------------------------------------------------

caminho_silvestre = (
    pasta_projeto
    / "data"
    / "processed"
    / "ProgramaRodofaunaAtropelada_silvestre.xlsx"
)

df_silvestre.to_excel(
    caminho_silvestre,
    index=False
)

print("Dataset silvestre salvo em:")
print(caminho_silvestre)



# %% ---------------------------------------------------------
# 34. VERIFICAÇÃO DO ARQUIVO SILVESTRE SALVO
# -------------------------------------------------------------

df_silvestre_verificacao = pd.read_excel(
    caminho_silvestre
)

assert len(df_silvestre_verificacao) == 4422

assert (
    df_silvestre_verificacao["silvestre_"]
    == "silvestre"
).all()

assert not (
    df_silvestre_verificacao["id"]
    .duplicated()
    .any()
)

print(
    "Arquivo silvestre salvo e verificado com sucesso."
)





# %% MISSÃO 5 - CONHECER AS COLUNAS DO DATASET SILVESTRE

print("\nCOLUNAS DO DATASET SILVESTRE")
print("--------------------------------")

for numero, coluna in enumerate(df_silvestre.columns, start=1):
    print(f"{numero:02d}. {coluna}")


print("\nTIPO DE DADO DE CADA COLUNA")
print("--------------------------------")
print(df_silvestre.dtypes)


print("\nQUANTIDADE DE VALORES AUSENTES")
print("--------------------------------")

valores_ausentes = (
    df_silvestre
    .isna()
    .sum()
    .sort_values(ascending=False)
)

print(valores_ausentes)


print("\nDISTRIBUIÇÃO DA COLUNA CLASSE")
print("--------------------------------")

quantidade_por_classe = df_silvestre["classe"].value_counts(
    dropna=False
)

percentual_por_classe = df_silvestre["classe"].value_counts(
    normalize=True,
    dropna=False
).mul(100).round(2)

resumo_classes = pd.DataFrame({
    "quantidade": quantidade_por_classe,
    "percentual": percentual_por_classe
})

print(resumo_classes)





# %% MISSÃO 6 - DEFINIR AS ENTRADAS E O ALVO

# Criamos uma cópia para não alterar o dataset silvestre original
df_modelagem = df_silvestre.copy()

# Garantimos que a coluna seja interpretada como uma data
df_modelagem["data_dt"] = pd.to_datetime(
    df_modelagem["data_dt"],
    errors="coerce"
)

# Extraímos informações confiáveis diretamente da data
df_modelagem["ano_data"] = df_modelagem["data_dt"].dt.year
df_modelagem["mes_data"] = df_modelagem["data_dt"].dt.month
df_modelagem["dia_semana_data"] = df_modelagem["data_dt"].dt.dayofweek


# Colunas que serão apresentadas ao modelo como pistas
colunas_entrada = [
    "ano_data",
    "mes_data",
    "dia_semana_data",
    "uc",
    "x",
    "y"
]

# Coluna que o modelo tentará prever
coluna_alvo = "classe"

# Criamos uma tabela contendo somente o necessário
df_modelo_classe = df_modelagem[
    colunas_entrada + [coluna_alvo]
].copy()


print("\nTABELA INICIAL PARA MODELAGEM")
print("--------------------------------")

print(df_modelo_classe.head())

print("\nDimensões da tabela:")
print(f"Linhas: {df_modelo_classe.shape[0]}")
print(f"Colunas: {df_modelo_classe.shape[1]}")

print("\nColunas de entrada:")
for coluna in colunas_entrada:
    print(f"- {coluna}")

print(f"\nColuna alvo: {coluna_alvo}")

print("\nValores ausentes:")
print(df_modelo_classe.isna().sum())





# %% MISSÃO 7 - TRATAR VALORES AUSENTES

# Trabalhamos sobre uma cópia
df_modelo_classe_limpo = df_modelo_classe.copy()

# Padronizamos temporariamente o texto da classe para facilitar a busca
classe_normalizada = (
    df_modelo_classe_limpo["classe"]
    .astype("string")
    .str.strip()
    .str.lower()
)

# Localiza valores nulos ou textos como "nao_identificado"
classe_invalida = (
    classe_normalizada.isna()
    | classe_normalizada.str.contains("nao_identific", na=False)
    | classe_normalizada.str.contains("não identific", na=False)
)

print("\nREGISTROS SEM CLASSE VÁLIDA")
print("--------------------------------")
print(df_modelo_classe_limpo.loc[classe_invalida])

print("\nQuantidade encontrada:")
print(classe_invalida.sum())


quantidade_antes = len(df_modelo_classe_limpo)

df_modelo_classe_limpo = (
    df_modelo_classe_limpo
    .loc[~classe_invalida]
    .copy()
    .reset_index(drop=True)
)

quantidade_depois = len(df_modelo_classe_limpo)

print("\nTRATAMENTO DA CLASSE")
print("--------------------------------")
print(f"Linhas antes: {quantidade_antes}")
print(f"Linhas depois: {quantidade_depois}")
print(f"Linhas removidas: {quantidade_antes - quantidade_depois}")


df_modelo_classe_limpo["uc"] = (
    df_modelo_classe_limpo["uc"]
    .astype("string")
    .str.strip()
    .replace("", pd.NA)
    .fillna("nao_identificada")
)


print("\nVERIFICAÇÃO APÓS O TRATAMENTO")
print("--------------------------------")

print(f"Quantidade final de linhas: {len(df_modelo_classe_limpo)}")

print("\nValores ausentes:")
print(df_modelo_classe_limpo.isna().sum())

print("\nQuantidade por unidade de conservação:")
print(df_modelo_classe_limpo["uc"].value_counts(dropna=False))

print("\nQuantidade por classe:")
print(df_modelo_classe_limpo["classe"].value_counts(dropna=False))


assert len(df_modelo_classe_limpo) == 4421
assert df_modelo_classe_limpo["classe"].isna().sum() == 0
assert df_modelo_classe_limpo["uc"].isna().sum() == 0
assert df_modelo_classe_limpo.isna().sum().sum() == 0

print("\nMissão 7 concluída: tabela sem valores ausentes.")
# %%




# %% MISSÃO 8 - SEPARAR TREINO, VALIDAÇÃO E TESTE

from sklearn.model_selection import train_test_split

X = df_modelo_classe_limpo[colunas_entrada].copy()
y = df_modelo_classe_limpo[coluna_alvo].copy()

print("Formato de X:", X.shape)
print("Formato de y:", y.shape)


X_treino, X_temporario, y_treino, y_temporario = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


X_validacao, X_teste, y_validacao, y_teste = train_test_split(
    X_temporario,
    y_temporario,
    test_size=0.50,
    random_state=42,
    stratify=y_temporario
)


del X_temporario
del y_temporario


print("\nTAMANHO DOS CONJUNTOS")
print("--------------------------------")

print(f"Treino:    {len(X_treino)} registros")
print(f"Validação: {len(X_validacao)} registros")
print(f"Teste:     {len(X_teste)} registros")

print("\nPERCENTUAIS")
print("--------------------------------")

total = len(X)

print(f"Treino:    {len(X_treino) / total * 100:.2f}%")
print(f"Validação: {len(X_validacao) / total * 100:.2f}%")
print(f"Teste:     {len(X_teste) / total * 100:.2f}%")



def percentual_por_classe(serie):
    return (
        serie
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .sort_index()
    )


distribuicao_classes = pd.concat(
    {
        "Dataset completo": percentual_por_classe(y),
        "Treino": percentual_por_classe(y_treino),
        "Validação": percentual_por_classe(y_validacao),
        "Teste": percentual_por_classe(y_teste)
    },
    axis=1
)

print("\nPERCENTUAL DE CADA CLASSE")
print("--------------------------------")
print(distribuicao_classes)


# Confirma que nenhuma linha apareceu em dois conjuntos
assert set(X_treino.index).isdisjoint(X_validacao.index)
assert set(X_treino.index).isdisjoint(X_teste.index)
assert set(X_validacao.index).isdisjoint(X_teste.index)

# Confirma que nenhuma linha foi perdida
assert (
    len(X_treino)
    + len(X_validacao)
    + len(X_teste)
    == len(X)
)

# Confirma que entradas e respostas continuam alinhadas
assert X_treino.index.equals(y_treino.index)
assert X_validacao.index.equals(y_validacao.index)
assert X_teste.index.equals(y_teste.index)

print("\nMissão 8 concluída: divisão realizada corretamente.")
# %%





# %% MISSÃO 9 - PREPARAR AS ENTRADAS PARA OS MODELOS

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

colunas_numericas = [
    "ano_data",
    "x",
    "y"
]

colunas_categoricas = [
    "mes_data",
    "dia_semana_data",
    "uc"
]

print("Colunas numéricas:", colunas_numericas)
print("Colunas categóricas:", colunas_categoricas)



preprocessador = ColumnTransformer(
    transformers=[
        (
            "numericas",
            StandardScaler(),
            colunas_numericas
        ),
        (
            "categoricas",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            colunas_categoricas
        )
    ]
)


X_treino_preparado = preprocessador.fit_transform(X_treino)

X_validacao_preparado = preprocessador.transform(X_validacao)
X_teste_preparado = preprocessador.transform(X_teste)


nomes_colunas_preparadas = (
    preprocessador.get_feature_names_out()
)

print("\nCOLUNAS APÓS A PREPARAÇÃO")
print("--------------------------------")

for numero, coluna in enumerate(
    nomes_colunas_preparadas,
    start=1
):
    print(f"{numero:02d}. {coluna}")
    
    
    
X_treino_preparado = pd.DataFrame(
    X_treino_preparado,
    columns=nomes_colunas_preparadas,
    index=X_treino.index
)

X_validacao_preparado = pd.DataFrame(
    X_validacao_preparado,
    columns=nomes_colunas_preparadas,
    index=X_validacao.index
)

X_teste_preparado = pd.DataFrame(
    X_teste_preparado,
    columns=nomes_colunas_preparadas,
    index=X_teste.index
)


print("\nFORMATOS APÓS A PREPARAÇÃO")
print("--------------------------------")

print("Treino:", X_treino_preparado.shape)
print("Validação:", X_validacao_preparado.shape)
print("Teste:", X_teste_preparado.shape)

print("\nPRIMEIRAS LINHAS DO TREINO")
print("--------------------------------")
print(X_treino_preparado.head())


assert len(X_treino_preparado) == len(y_treino)
assert len(X_validacao_preparado) == len(y_validacao)
assert len(X_teste_preparado) == len(y_teste)

assert not X_treino_preparado.isna().any().any()
assert not X_validacao_preparado.isna().any().any()
assert not X_teste_preparado.isna().any().any()

assert list(X_treino_preparado.columns) == list(
    X_validacao_preparado.columns
)

assert list(X_treino_preparado.columns) == list(
    X_teste_preparado.columns
)

print("\nMissão 9 concluída: entradas preparadas corretamente.")



# %%

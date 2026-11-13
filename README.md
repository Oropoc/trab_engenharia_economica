# Pipeline de análise de atropelamentos de fauna

Projeto acadêmico com registros do Programa RodoFauna. O pipeline prepara os
dados, explora padrões, compara modelos de classificação, estima contagens por
célula espacial e identifica áreas com maior concentração de registros.

## Como executar

1. Confirme que o Excel original está em
   `data/raw/ProgramaRodofaunaAtropelada_original.xlsx`.
2. Instale as bibliotecas listadas em `requirements.txt` no Python que será
   usado para rodar o projeto.
3. Execute `rodar_tudo.bat`.

O arquivo `.bat` está configurado para o ambiente desta máquina
(`C:\Python314\python.exe` e `C:\ai_env\site-packages`). Se o Python ou as
bibliotecas estiverem em outro local, ajuste `PYTHON_EXE` e `PYTHONPATH` no
início do `.bat`. Também é possível executar os scripts individualmente, na
ordem em que aparecem no arquivo.

## Etapas

1. `scripts/01_preparacao_dados.py`: audita IDs, remove duplicatas idênticas,
   mantém registros silvestres e prepara o dataset para análise.
2. `scripts/02_analise_exploratoria.py`: resume dados, classes, datas e
   coordenadas; grava gráficos em `resultados/figuras`.
3. `scripts/03_classificacao_multiclasse.py`: executa os baselines iniciais.
4. `scripts/04_classificacao_otimizada.py`: compara Dummy, regressão
   logística, Random Forest e MLP. Faz busca em grade e validação cruzada
   estratificada no conjunto de treino; usa o teste reservado apenas para
   avaliação. A diferença entre o F1-macro médio da validação cruzada e o do
   teste ajuda a examinar generalização.
5. `scripts/05_regressao_contagem.py`: agrega registros em células de 1 km e
   prevê a contagem anual registrada apenas em células com registros no
   período de treino. Células que só aparecem em anos posteriores não entram
   na previsão. Usa anos iniciais para treino, o ano
   anterior ao mais recente para validação e o ano mais recente como teste
   temporal. Compara Dummy, Poisson, Random Forest e MLP; otimiza os dois
   últimos usando grupos separados por ano.
6. `scripts/06_hotspots_espaciais.py`: calcula a quantidade de registros por
   célula de 1 km e marca como hotspots as células com contagem igual ou maior
   ao percentil 90 das células ocupadas. Empates no limiar podem fazer com que
   mais de 10% das células sejam marcadas.

## Principais resultados gerados

- `resultados/classificacao_ajustada.csv`: métricas de validação cruzada,
  teste reservado e hiperparâmetros selecionados.
- `resultados/relatorios_classificacao_teste.txt`: precisão, recall e F1 por
  classe no teste.
- `resultados/regressao_contagem_modelos.csv`: comparação no ano de validação.
- `resultados/previsoes_contagem_teste.csv`: previsões do modelo escolhido
  pela validação para o ano de teste.
- `resultados/hotspots_grade_1000m.csv`: células, contagens e indicador de
  hotspot, com centros e índices espaciais.
- `resultados/hotspots_grade_1000m.png`: mapa dos pontos e células marcadas.

## Interpretação e limitações

- A classificação prevê a classe taxonômica de um animal que já foi
  registrado; não prevê se ocorrerá um atropelamento.
- A regressão prevê número de registros por célula/ano para células conhecidas
  no treino. Célula/ano sem registro recebe contagem zero, o que significa
  “nenhum registro disponível no arquivo”, não prova de ausência de
  atropelamentos.
- A grade usa coordenadas em unidades presumidas como metros. Confirme o
  sistema de referência espacial (CRS) nos metadados do serviço do IBRAM antes
  de interpretar distâncias ou sobrepor os CSVs em um SIG.
- O arquivo não contém, por si só, esforço de busca por trecho e período,
  fluxo de veículos ou cobertura de todas as rodovias. Portanto, contagens e
  hotspots são concentrações de ocorrências observadas, não estimativas de
  risco absoluto. Uma análise de risco exigiria esses dados de exposição e
  observações de ausência.
- A validação da classificação usa amostra estratificada aleatória; a
  regressão usa separação temporal. O teste não deve ser usado para escolher
  hiperparâmetros ou modelos.

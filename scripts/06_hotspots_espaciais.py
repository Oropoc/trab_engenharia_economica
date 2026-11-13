"""Identifica células com maior concentração de registros de atropelamento."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle


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
PERCENTIL_HOTSPOT = 0.90

if not ARQUIVO_LIMPO.exists():
    raise FileNotFoundError(
        "Dataset limpo não encontrado. Execute primeiro "
        "scripts/01_preparacao_dados.py."
    )

df = pd.read_excel(ARQUIVO_LIMPO)
if df[["x", "y"]].isna().any().any():
    raise ValueError("O dataset contém coordenadas ausentes.")

df["cell_x"] = (df["x"] // TAMANHO_CELULA_METROS).astype(int)
df["cell_y"] = (df["y"] // TAMANHO_CELULA_METROS).astype(int)

grade = (
    df.groupby(["cell_x", "cell_y"])
    .size()
    .rename("registros")
    .reset_index()
)
limiar = float(grade["registros"].quantile(PERCENTIL_HOTSPOT))
grade["hotspot"] = grade["registros"] >= limiar
grade["x_centro_m"] = (
    grade["cell_x"] + 0.5
) * TAMANHO_CELULA_METROS
grade["y_centro_m"] = (
    grade["cell_y"] + 0.5
) * TAMANHO_CELULA_METROS
grade["tamanho_celula_m"] = TAMANHO_CELULA_METROS
grade["limiar_percentil_90"] = limiar

arquivo_grade = PASTA_RESULTADOS / "hotspots_grade_1000m.csv"
grade.sort_values("registros", ascending=False).to_csv(
    arquivo_grade,
    index=False,
    encoding="utf-8-sig",
)

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(
    df["x"],
    df["y"],
    s=5,
    alpha=0.25,
    color="#65727a",
    label="Registros",
)
for celula in grade.loc[grade["hotspot"]].itertuples():
    ax.add_patch(
        Rectangle(
            (celula.cell_x * TAMANHO_CELULA_METROS,
             celula.cell_y * TAMANHO_CELULA_METROS),
            TAMANHO_CELULA_METROS,
            TAMANHO_CELULA_METROS,
            facecolor="#e63946",
            edgecolor="#9d0208",
            alpha=0.25,
        )
    )
ax.scatter(
    grade.loc[grade["hotspot"], "x_centro_m"],
    grade.loc[grade["hotspot"], "y_centro_m"],
    s=20,
    color="#b00020",
    label="Células hotspot",
)
ax.set_title("Concentração de registros por célula de 1 km")
ax.set_xlabel("Coordenada X (unidade do dataset)")
ax.set_ylabel("Coordenada Y (unidade do dataset)")
ax.legend()
ax.set_aspect("equal", adjustable="datalim")
fig.tight_layout()

arquivo_mapa = PASTA_RESULTADOS / "hotspots_grade_1000m.png"
fig.savefig(arquivo_mapa, dpi=160)
plt.close(fig)

print(f"Total de células com ao menos um registro: {len(grade)}")
print(
    f"Limiar do percentil 90: {limiar:.0f} registros por célula; "
    f"células marcadas: {int(grade['hotspot'].sum())}."
)
print(f"Tabela da grade salva em: {arquivo_grade}")
print(f"Mapa salvo em: {arquivo_mapa}")
print(
    "\nInterpretação: hotspots representam concentração de registros "
    "observados, não risco absoluto. A grade não considera quilômetros "
    "percorridos, esforço de busca, tráfego nem áreas não amostradas."
)

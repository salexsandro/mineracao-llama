"""
Gera visualizações estatísticas da relação entre sintomas psicológicos/cognitivos
e tópicos de discussão extraídos via BERTopic.

Entrada: arquivo JSONL produzido por merge_symptoms_topics.py
Saída:   pasta graficos/ com todos os gráficos em PNG

Sintomas com menos de MIN_SYMPTOM_COUNT posts no total são descartados
de TODAS as análises, para evitar poluição visual com categorias raras.

Biblioteca: seaborn sobre matplotlib — padrão em papers de NLP/saúde mental
(ex: Yang et al. 2023, MentalLLaMA) para heatmaps e gráficos estatísticos,
pela legibilidade superior em anotações automáticas e paletas perceptualmente
uniformes.

Análises geradas:
  1. Histogramas por tópico       (contagem absoluta e percentual)
  2. Histogramas por sintoma      (contagem absoluta e percentual)
  3. Histogramas por macrotópico  (contagem absoluta e percentual)
  4. Heatmap tópico × sintoma
  5. Heatmap macrotópico × sintoma
  6. Co-ocorrência de sintomas
  7. Distribuição de confiança por tópico (boxplot)
  8. Diversidade de sintomas por tópico (entropia de Shannon)
"""

import json
import os
import sys
import io
import logging
import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ─────────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────────

INPUT_FILE = "posts_topics_symptoms_merged.jsonl"
OUTPUT_DIR = "graficos"

# Sintomas com menos posts que isso (somando todas as ocorrências no corpus)
# são descartados de TODAS as análises.
MIN_SYMPTOM_COUNT = 50

# Mapeamento tópico → macrotópico, baseado na sumarização fornecida
MACRO_THEMES = {
    "Engajamento Digital Compulsivo":        [0, 9],
    "Autorregulação e Desintoxicação Digital": [3, 7, 8],
    "Produtividade e Desempenho Cognitivo":  [2],
    "Saúde Mental e Condições Neuropsicológicas": [4, 6],
    "Transições de Vida e Ajuste Social":    [1],
    "Crítica a Plataformas e Impacto Social": [5],
}

TOPIC_TO_MACRO = {
    topic_id: macro
    for macro, topic_ids in MACRO_THEMES.items()
    for topic_id in topic_ids
}

TOPIC_LABELS_PT = {
    0: "Doomscrolling e sono",
    1: "Transições de vida",
    2: "Procrastinação acadêmica",
    3: "Gestão de tempo de tela",
    4: "Ansiedade e consumo de mídia",
    5: "Crítica ao TikTok",
    6: "Diagnóstico e medicação TDAH",
    7: "Redução de redes sociais",
    8: "Detox de dopamina",
    9: "Uso compulsivo aprisionante",
}

DPI = 150
sns.set_theme(style="white", font_scale=1.0)

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

_stream_handler = logging.StreamHandler(
    stream=io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[_stream_handler]
)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.titlesize": 15,
})


# =============================================================================
# CARREGAMENTO, FILTRAGEM E PREPARAÇÃO DOS DADOS
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    df = pd.DataFrame(records)
    logging.info(f"Carregados {len(df)} posts de {path}")

    df["Symptom"] = df["Symptom"].apply(lambda x: x if isinstance(x, list) else [])

    n_before = len(df)
    df = df[df["Symptom"].apply(len) > 0].copy()
    logging.info(f"Removidos {n_before - len(df)} posts sem sintoma atribuído")

    df["topic_label_pt"] = df["topic_id"].map(TOPIC_LABELS_PT).fillna(
        df["topic_id"].astype(str)
    )
    df["macro_theme"] = df["topic_id"].map(TOPIC_TO_MACRO).fillna("Não classificado")

    return df


def filter_rare_symptoms(df: pd.DataFrame, min_count: int) -> pd.DataFrame:
    """
    Remove sintomas com menos de `min_count` ocorrências totais no corpus.
    Posts que ficam sem nenhum sintoma após a filtragem são descartados
    das análises (reportado no log).
    """
    all_symptoms = [s for lst in df["Symptom"] for s in lst]
    symptom_counts = pd.Series(all_symptoms).value_counts()

    kept_symptoms = set(symptom_counts[symptom_counts >= min_count].index)
    dropped_symptoms = set(symptom_counts[symptom_counts < min_count].index)

    logging.info(f"\nFiltragem de sintomas raros (mínimo {min_count} posts):")
    logging.info(f"  Sintomas mantidos ({len(kept_symptoms)}): {sorted(kept_symptoms)}")
    if dropped_symptoms:
        dropped_info = [
            f"{s} (n={symptom_counts[s]})" for s in sorted(dropped_symptoms)
        ]
        logging.info(f"  Sintomas descartados ({len(dropped_symptoms)}): {', '.join(dropped_info)}")
    else:
        logging.info("  Nenhum sintoma descartado.")

    df = df.copy()
    df["Symptom"] = df["Symptom"].apply(
        lambda lst: [s for s in lst if s in kept_symptoms]
    )

    n_before = len(df)
    df = df[df["Symptom"].apply(len) > 0].copy()
    n_dropped_posts = n_before - len(df)
    if n_dropped_posts > 0:
        logging.info(
            f"  {n_dropped_posts} posts descartados por terem apenas "
            f"sintomas raros (ficaram sem nenhum sintoma válido)."
        )

    return df


def explode_symptoms(df: pd.DataFrame) -> pd.DataFrame:
    return df.explode("Symptom").rename(columns={"Symptom": "symptom"})


# =============================================================================
# 1. HISTOGRAMAS POR TÓPICO
# =============================================================================

def plot_symptoms_per_topic(df_long: pd.DataFrame, output_dir: str):
    counts = (
        df_long.groupby(["topic_label_pt", "symptom"])
        .size()
        .reset_index(name="count")
    )

    pivot_count = counts.pivot(index="topic_label_pt", columns="symptom", values="count").fillna(0)
    topic_order = pivot_count.sum(axis=1).sort_values(ascending=False).index
    pivot_count = pivot_count.loc[topic_order]
    pivot_pct = pivot_count.div(pivot_count.sum(axis=1), axis=0) * 100

    n_symptoms = pivot_count.shape[1]
    palette = sns.color_palette("Dark2", n_symptoms)

    # ── Gráfico 1: contagem absoluta ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, max(6, 0.5 * len(pivot_count))))
    pivot_count.plot(kind="barh", stacked=True, ax=ax, color=palette,
                      width=0.7, edgecolor="white", linewidth=0.6)
    ax.set_title("Frequência Absoluta de Sintomas por Tópico", pad=15)
    ax.set_ylabel("")
    ax.set_xlabel("Número de Posts")
    ax.legend(title="Sintoma", bbox_to_anchor=(1.02, 1), loc="upper left",
               fontsize=9, frameon=False)
    ax.invert_yaxis()
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01a_sintomas_por_topico_absoluto.png"), dpi=DPI)
    plt.close()

    # ── Gráfico 2: percentual ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, max(6, 0.5 * len(pivot_count))))
    pivot_pct.plot(kind="barh", stacked=True, ax=ax, color=palette,
                    width=0.7, edgecolor="white", linewidth=0.6)
    ax.set_title("Distribuição Percentual de Sintomas por Tópico", pad=15)
    ax.set_ylabel("")
    ax.set_xlabel("Percentual de Posts (%)")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(title="Sintoma", bbox_to_anchor=(1.02, 1), loc="upper left",
               fontsize=9, frameon=False)
    ax.invert_yaxis()
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01b_sintomas_por_topico_percentual.png"), dpi=DPI)
    plt.close()

    # ── Gráfico 3: sintoma DOMINANTE por tópico ───────────────────────────────
    dominant = pivot_pct.idxmax(axis=1)
    dominant_pct = pivot_pct.max(axis=1)
    order = dominant_pct.sort_values(ascending=True).index

    fig, ax = plt.subplots(figsize=(11, max(5, 0.5 * len(order))))
    bars = ax.barh(
        [f"{t}  —  {dominant[t]}" for t in order],
        dominant_pct.loc[order],
        color=sns.color_palette("flare", len(order)),
        edgecolor="white"
    )
    ax.set_title("Sintoma Mais Comum em Cada Tópico", pad=15)
    ax.set_xlabel("Percentual de Posts com o Sintoma Dominante (%)")
    for bar, val in zip(bars, dominant_pct.loc[order]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=9)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01c_sintoma_dominante_por_topico.png"), dpi=DPI)
    plt.close()

    logging.info("  [1/8] Histogramas por tópico salvos.")
    return pivot_count, pivot_pct


# =============================================================================
# 2. HISTOGRAMAS POR SINTOMA
# =============================================================================

def plot_topics_per_symptom(df_long: pd.DataFrame, output_dir: str):
    counts = (
        df_long.groupby(["symptom", "topic_label_pt"])
        .size()
        .reset_index(name="count")
    )

    pivot_count = counts.pivot(index="symptom", columns="topic_label_pt", values="count").fillna(0)
    symptom_order = pivot_count.sum(axis=1).sort_values(ascending=False).index
    pivot_count = pivot_count.loc[symptom_order]
    pivot_pct = pivot_count.div(pivot_count.sum(axis=1), axis=0) * 100

    n_topics = pivot_count.shape[1]
    palette = sns.color_palette("Dark2", n_topics)

    # ── Gráfico 1: contagem absoluta ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, max(5, 0.6 * len(pivot_count))))
    pivot_count.plot(kind="barh", stacked=True, ax=ax, color=palette,
                      width=0.7, edgecolor="white", linewidth=0.6)
    ax.set_title("Frequência Absoluta de Tópicos por Sintoma", pad=15)
    ax.set_ylabel("")
    ax.set_xlabel("Número de Posts")
    ax.legend(title="Tópico", bbox_to_anchor=(1.02, 1), loc="upper left",
               fontsize=8, frameon=False)
    ax.invert_yaxis()
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02a_topicos_por_sintoma_absoluto.png"), dpi=DPI)
    plt.close()

    # ── Gráfico 2: percentual ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, max(5, 0.6 * len(pivot_count))))
    pivot_pct.plot(kind="barh", stacked=True, ax=ax, color=palette,
                    width=0.7, edgecolor="white", linewidth=0.6)
    ax.set_title("Distribuição Percentual de Tópicos por Sintoma", pad=15)
    ax.set_ylabel("")
    ax.set_xlabel("Percentual de Posts (%)")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(title="Tópico", bbox_to_anchor=(1.02, 1), loc="upper left",
               fontsize=8, frameon=False)
    ax.invert_yaxis()
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02b_topicos_por_sintoma_percentual.png"), dpi=DPI)
    plt.close()

    # ── Gráfico 3: tópico DOMINANTE por sintoma ───────────────────────────────
    dominant = pivot_pct.idxmax(axis=1)
    dominant_pct = pivot_pct.max(axis=1)
    order = dominant_pct.sort_values(ascending=True).index

    fig, ax = plt.subplots(figsize=(11, max(5, 0.5 * len(order))))
    bars = ax.barh(
        [f"{s}  —  {dominant[s]}" for s in order],
        dominant_pct.loc[order],
        color=sns.color_palette("flare", len(order)),
        edgecolor="white"
    )
    ax.set_title("Tópico Mais Associado a Cada Sintoma", pad=15)
    ax.set_xlabel("Percentual de Posts no Tópico Dominante (%)")
    for bar, val in zip(bars, dominant_pct.loc[order]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=9)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02c_topico_dominante_por_sintoma.png"), dpi=DPI)
    plt.close()

    logging.info("  [2/8] Histogramas por sintoma salvos.")
    return pivot_count, pivot_pct


# =============================================================================
# 3. HISTOGRAMAS POR MACROTÓPICO
# =============================================================================

def plot_symptoms_per_macro(df_long: pd.DataFrame, output_dir: str):
    counts = (
        df_long.groupby(["macro_theme", "symptom"])
        .size()
        .reset_index(name="count")
    )

    pivot_count = counts.pivot(index="macro_theme", columns="symptom", values="count").fillna(0)
    macro_order = pivot_count.sum(axis=1).sort_values(ascending=False).index
    pivot_count = pivot_count.loc[macro_order]
    pivot_pct = pivot_count.div(pivot_count.sum(axis=1), axis=0) * 100

    n_symptoms = pivot_count.shape[1]
    palette = sns.color_palette("Dark2", n_symptoms)

    # ── Gráfico 1: contagem absoluta ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, max(5, 0.7 * len(pivot_count))))
    pivot_count.plot(kind="barh", stacked=True, ax=ax, color=palette,
                      width=0.6, edgecolor="white", linewidth=0.6)
    ax.set_title("Frequência Absoluta de Sintomas por Macrotópico", pad=15)
    ax.set_ylabel("")
    ax.set_xlabel("Número de Posts")
    ax.legend(title="Sintoma", bbox_to_anchor=(1.02, 1), loc="upper left",
               fontsize=9, frameon=False)
    ax.invert_yaxis()
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03a_sintomas_por_macrotopico_absoluto.png"), dpi=DPI)
    plt.close()

    # ── Gráfico 2: percentual ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, max(5, 0.7 * len(pivot_count))))
    pivot_pct.plot(kind="barh", stacked=True, ax=ax, color=palette,
                    width=0.6, edgecolor="white", linewidth=0.6)
    ax.set_title("Distribuição Percentual de Sintomas por Macrotópico", pad=15)
    ax.set_ylabel("")
    ax.set_xlabel("Percentual de Posts (%)")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(title="Sintoma", bbox_to_anchor=(1.02, 1), loc="upper left",
               fontsize=9, frameon=False)
    ax.invert_yaxis()
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03b_sintomas_por_macrotopico_percentual.png"), dpi=DPI)
    plt.close()

    logging.info("  [3/8] Histogramas por macrotópico salvos.")
    return pivot_count, pivot_pct


# =============================================================================
# 4. HEATMAP TÓPICO × SINTOMA
# =============================================================================

def plot_heatmap_topic_symptom(df_long: pd.DataFrame, output_dir: str):
    counts = (
        df_long.groupby(["topic_label_pt", "symptom"])
        .size()
        .reset_index(name="count")
    )
    pivot = counts.pivot(index="topic_label_pt", columns="symptom", values="count").fillna(0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    order = pivot.sum(axis=1).sort_values(ascending=False).index
    pivot_pct = pivot_pct.loc[order]

    fig, ax = plt.subplots(figsize=(max(9, 1.1 * pivot_pct.shape[1]),
                                     max(7, 0.6 * pivot_pct.shape[0])))
    sns.heatmap(
        pivot_pct, annot=True, fmt=".0f", cmap="YlOrRd",
        cbar_kws={"label": "% de posts do tópico", "shrink": 0.8},
        linewidths=0.5, linecolor="white",
        annot_kws={"fontsize": 9}, ax=ax
    )
    ax.set_title("Mapa de Calor: Percentual de Sintomas por Tópico", pad=15)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=40, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_heatmap_topico_sintoma.png"), dpi=DPI)
    plt.close()

    logging.info("  [4/8] Heatmap tópico × sintoma salvo.")


# =============================================================================
# 5. HEATMAP MACROTÓPICO × SINTOMA
# =============================================================================

def plot_heatmap_macro_symptom(df_long: pd.DataFrame, output_dir: str):
    counts = (
        df_long.groupby(["macro_theme", "symptom"])
        .size()
        .reset_index(name="count")
    )
    pivot = counts.pivot(index="macro_theme", columns="symptom", values="count").fillna(0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    order = pivot.sum(axis=1).sort_values(ascending=False).index
    pivot_pct = pivot_pct.loc[order]

    fig, ax = plt.subplots(figsize=(max(9, 1.1 * pivot_pct.shape[1]),
                                     max(6, 0.8 * pivot_pct.shape[0])))
    sns.heatmap(
        pivot_pct, annot=True, fmt=".0f", cmap="YlOrRd",
        cbar_kws={"label": "% de posts do macrotópico", "shrink": 0.8},
        linewidths=0.5, linecolor="white",
        annot_kws={"fontsize": 10}, ax=ax
    )
    ax.set_title("Mapa de Calor: Percentual de Sintomas por Macrotópico", pad=15)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=40, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_heatmap_macrotopico_sintoma.png"), dpi=DPI)
    plt.close()

    logging.info("  [5/8] Heatmap macrotópico × sintoma salvo.")


# =============================================================================
# 6. CO-OCORRÊNCIA DE SINTOMAS
# =============================================================================

def plot_symptom_cooccurrence(df: pd.DataFrame, output_dir: str):
    all_symptoms = sorted(set(s for lst in df["Symptom"] for s in lst))
    co_matrix = pd.DataFrame(0, index=all_symptoms, columns=all_symptoms, dtype=int)

    for symptoms_list in df["Symptom"]:
        unique_symptoms = sorted(set(symptoms_list))
        for s1, s2 in itertools.combinations(unique_symptoms, 2):
            co_matrix.loc[s1, s2] += 1
            co_matrix.loc[s2, s1] += 1
        for s in unique_symptoms:
            co_matrix.loc[s, s] += 1

    # Máscara para o triângulo superior — evita redundância visual
    mask = np.triu(np.ones_like(co_matrix, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(all_symptoms)),
                                     max(7, 0.8 * len(all_symptoms))))
    sns.heatmap(
        co_matrix, mask=mask, annot=True, fmt="d", cmap="PuBu",
        cbar_kws={"label": "Número de posts", "shrink": 0.8},
        linewidths=0.5, linecolor="white",
        annot_kws={"fontsize": 9}, ax=ax, square=True
    )
    ax.set_title("Co-ocorrência de Sintomas no Mesmo Post\n(diagonal = total de posts com o sintoma)", pad=15)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=40, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_coocorrencia_sintomas.png"), dpi=DPI)
    plt.close()

    logging.info("  [6/8] Matriz de co-ocorrência de sintomas salva.")


# =============================================================================
# 7. DISTRIBUIÇÃO DE CONFIANÇA POR TÓPICO
# =============================================================================

def plot_confidence_by_topic(df: pd.DataFrame, output_dir: str):
    if "confidence" not in df.columns:
        logging.warning("  Coluna 'confidence' ausente — pulando gráfico de confiança.")
        return

    order = (
        df.groupby("topic_label_pt")["confidence"]
        .median()
        .sort_values(ascending=False)
        .index
    )

    fig, ax = plt.subplots(figsize=(11, max(6, 0.55 * len(order))))
    sns.boxplot(
        data=df, y="topic_label_pt", x="confidence", order=order,
        hue="topic_label_pt", palette="crest", showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white",
                   "markeredgecolor": "black", "markersize": 6},
        legend=False, ax=ax
    )
    ax.set_title("Distribuição da Confiança de Atribuição por Tópico", pad=15)
    ax.set_xlabel("Confiança da Atribuição (softmax)")
    ax.set_ylabel("")
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "07_confianca_por_topico.png"), dpi=DPI)
    plt.close()

    logging.info("  [7/8] Distribuição de confiança por tópico salva.")


# =============================================================================
# 8. DIVERSIDADE DE SINTOMAS POR TÓPICO (ENTROPIA DE SHANNON)
# =============================================================================

def shannon_entropy(counts: np.ndarray) -> float:
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def plot_symptom_diversity(pivot_count: pd.DataFrame, output_dir: str):
    entropies = pivot_count.apply(lambda row: shannon_entropy(row.values), axis=1)
    max_entropy = np.log2(pivot_count.shape[1])
    entropies_norm = entropies / max_entropy
    order = entropies_norm.sort_values(ascending=True).index

    fig, ax = plt.subplots(figsize=(11, max(6, 0.55 * len(order))))
    bars = ax.barh(order, entropies_norm.loc[order],
                    color=sns.color_palette("mako", len(order)),
                    edgecolor="white")
    ax.set_title("Diversidade de Sintomas por Tópico (Entropia de Shannon Normalizada)", pad=15)
    ax.set_xlabel("Entropia Normalizada (0 = um sintoma domina; 1 = sintomas equilibrados)")
    ax.set_xlim(0, 1.05)
    for bar, val in zip(bars, entropies_norm.loc[order]):
        ax.text(val + 0.015, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=9)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "08_diversidade_sintomas_por_topico.png"), dpi=DPI)
    plt.close()

    logging.info("  [8/8] Diversidade de sintomas (entropia) salva.")


# =============================================================================
# RELATÓRIO TXT — NÚMEROS E PERCENTUAIS DETALHADOS
# =============================================================================

def write_report(df: pd.DataFrame, df_long: pd.DataFrame, output_path: str,
                  min_symptom_count: int, n_posts_original: int):
    """
    Gera um relatório textual completo com:
      - Resumo geral do corpus (antes/depois da filtragem)
      - Tabela: sintomas por tópico (contagem e %)
      - Tabela: tópicos por sintoma (contagem e %)
      - Tabela: sintomas por macrotópico (contagem e %)
      - Ranking de sintomas mais/menos comuns globalmente
      - Ranking de tópicos por volume de posts
    """
    lines = []
    W = 78

    def header(title):
        lines.append("")
        lines.append("=" * W)
        lines.append(f"  {title}")
        lines.append("=" * W)

    def subheader(title):
        lines.append("")
        lines.append("-" * W)
        lines.append(f"  {title}")
        lines.append("-" * W)

    # ── Resumo geral ──────────────────────────────────────────────────────────
    header("RELATÓRIO DE ANÁLISE — SINTOMAS × TÓPICOS")
    lines.append(f"  Posts no arquivo de entrada       : {n_posts_original}")
    lines.append(f"  Posts após filtragem               : {len(df)}")
    lines.append(f"  Limiar mínimo de sintoma (posts)   : {min_symptom_count}")
    lines.append(f"  Sintomas analisados                : {sorted(df_long['symptom'].unique())}")
    lines.append(f"  Total de tópicos                   : {df['topic_label_pt'].nunique()}")
    lines.append(f"  Total de macrotópicos               : {df['macro_theme'].nunique()}")
    lines.append(f"  Total de atribuições sintoma-post   : {len(df_long)}")
    lines.append(f"  Média de sintomas por post          : {len(df_long)/len(df):.2f}")

    # ── 1. Sintomas por Tópico ───────────────────────────────────────────────
    header("1. SINTOMAS POR TÓPICO")

    counts_t = (
        df_long.groupby(["topic_label_pt", "symptom"])
        .size().reset_index(name="count")
    )
    pivot_t = counts_t.pivot(index="topic_label_pt", columns="symptom", values="count").fillna(0).astype(int)
    pivot_t_pct = pivot_t.div(pivot_t.sum(axis=1), axis=0) * 100
    topic_totals = pivot_t.sum(axis=1).sort_values(ascending=False)

    for topic in topic_totals.index:
        total = topic_totals[topic]
        lines.append("")
        lines.append(f"  {topic}  (total: {total} ocorrências de sintomas)")
        lines.append("  " + "-" * 60)
        row_counts = pivot_t.loc[topic].sort_values(ascending=False)
        row_pcts   = pivot_t_pct.loc[topic]
        for symptom, count in row_counts.items():
            if count == 0:
                continue
            pct = row_pcts[symptom]
            lines.append(f"    {symptom:<22} {count:>6} posts   ({pct:>5.1f}%)")

    # ── 2. Tópicos por Sintoma ───────────────────────────────────────────────
    header("2. TÓPICOS POR SINTOMA")

    counts_s = (
        df_long.groupby(["symptom", "topic_label_pt"])
        .size().reset_index(name="count")
    )
    pivot_s = counts_s.pivot(index="symptom", columns="topic_label_pt", values="count").fillna(0).astype(int)
    pivot_s_pct = pivot_s.div(pivot_s.sum(axis=1), axis=0) * 100
    symptom_totals = pivot_s.sum(axis=1).sort_values(ascending=False)

    for symptom in symptom_totals.index:
        total = symptom_totals[symptom]
        lines.append("")
        lines.append(f"  {symptom}  (total: {total} posts)")
        lines.append("  " + "-" * 60)
        row_counts = pivot_s.loc[symptom].sort_values(ascending=False)
        row_pcts   = pivot_s_pct.loc[symptom]
        for topic, count in row_counts.items():
            if count == 0:
                continue
            pct = row_pcts[topic]
            lines.append(f"    {topic:<32} {count:>6} posts   ({pct:>5.1f}%)")

    # ── 3. Sintomas por Macrotópico ──────────────────────────────────────────
    header("3. SINTOMAS POR MACROTÓPICO")

    counts_m = (
        df_long.groupby(["macro_theme", "symptom"])
        .size().reset_index(name="count")
    )
    pivot_m = counts_m.pivot(index="macro_theme", columns="symptom", values="count").fillna(0).astype(int)
    pivot_m_pct = pivot_m.div(pivot_m.sum(axis=1), axis=0) * 100
    macro_totals = pivot_m.sum(axis=1).sort_values(ascending=False)

    for macro in macro_totals.index:
        total = macro_totals[macro]
        n_posts_macro = df[df["macro_theme"] == macro].shape[0]
        lines.append("")
        lines.append(f"  {macro}  ({n_posts_macro} posts, {total} ocorrências de sintomas)")
        lines.append("  " + "-" * 60)
        row_counts = pivot_m.loc[macro].sort_values(ascending=False)
        row_pcts   = pivot_m_pct.loc[macro]
        for symptom, count in row_counts.items():
            if count == 0:
                continue
            pct = row_pcts[symptom]
            lines.append(f"    {symptom:<22} {count:>6} posts   ({pct:>5.1f}%)")

    # ── 4. Ranking global de sintomas ────────────────────────────────────────
    header("4. RANKING GLOBAL DE SINTOMAS  (todo o corpus filtrado)")
    total_posts = len(df)
    lines.append("")
    hdr = f"  {'Rank':>4}  {'Sintoma':<22} {'Posts':>8}  {'% do corpus':>12}"
    lines.append(hdr)
    lines.append("  " + "-" * 50)
    for rank, (symptom, count) in enumerate(symptom_totals.items(), 1):
        pct = count / total_posts * 100
        lines.append(f"  {rank:>4}  {symptom:<22} {count:>8}  {pct:>11.1f}%")

    # ── 5. Ranking de tópicos por volume ─────────────────────────────────────
    header("5. RANKING DE TÓPICOS POR VOLUME DE POSTS")
    topic_post_counts = df["topic_label_pt"].value_counts()
    lines.append("")
    hdr = f"  {'Rank':>4}  {'Tópico':<32} {'Posts':>8}  {'% do corpus':>12}"
    lines.append(hdr)
    lines.append("  " + "-" * 60)
    for rank, (topic, count) in enumerate(topic_post_counts.items(), 1):
        pct = count / total_posts * 100
        lines.append(f"  {rank:>4}  {topic:<32} {count:>8}  {pct:>11.1f}%")

    # ── 6. Sintoma dominante por tópico (resumo rápido) ──────────────────────
    header("6. RESUMO — SINTOMA DOMINANTE POR TÓPICO")
    dominant_by_topic = pivot_t_pct.idxmax(axis=1)
    dominant_pct_by_topic = pivot_t_pct.max(axis=1)
    lines.append("")
    for topic in topic_totals.index:
        lines.append(
            f"  {topic:<32} → {dominant_by_topic[topic]:<20} "
            f"({dominant_pct_by_topic[topic]:.1f}% dos posts do tópico)"
        )

    # ── 7. Tópico dominante por sintoma (resumo rápido) ──────────────────────
    header("7. RESUMO — TÓPICO DOMINANTE POR SINTOMA")
    dominant_by_symptom = pivot_s_pct.idxmax(axis=1)
    dominant_pct_by_symptom = pivot_s_pct.max(axis=1)
    lines.append("")
    for symptom in symptom_totals.index:
        lines.append(
            f"  {symptom:<22} → {dominant_by_symptom[symptom]:<32} "
            f"({dominant_pct_by_symptom[symptom]:.1f}% dos posts do sintoma)"
        )

    lines.append("")
    lines.append("=" * W)
    lines.append("  FIM DO RELATÓRIO")
    lines.append("=" * W)

    text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    logging.info(f"  Relatório TXT salvo: {output_path}")
    return text


# =============================================================================
# MAIN
# =============================================================================

def run(input_path: str, output_dir: str, min_symptom_count: int):
    os.makedirs(output_dir, exist_ok=True)

    df_original = load_data(input_path)
    n_posts_original = len(df_original)

    df = filter_rare_symptoms(df_original, min_symptom_count)
    df_long = explode_symptoms(df)

    logging.info(f"\nTotal de posts após filtragem: {len(df)}")
    logging.info(f"Total de atribuições sintoma-post (long format): {len(df_long)}")
    logging.info(f"Sintomas finais analisados: {sorted(df_long['symptom'].unique())}")
    logging.info(f"\nGerando gráficos em '{output_dir}/'...\n")

    pivot_count_topic, _ = plot_symptoms_per_topic(df_long, output_dir)
    plot_topics_per_symptom(df_long, output_dir)
    plot_symptoms_per_macro(df_long, output_dir)
    plot_heatmap_topic_symptom(df_long, output_dir)
    plot_heatmap_macro_symptom(df_long, output_dir)
    plot_symptom_cooccurrence(df, output_dir)
    plot_confidence_by_topic(df, output_dir)
    plot_symptom_diversity(pivot_count_topic, output_dir)

    logging.info(f"\nGerando relatório TXT...")
    report_path = os.path.join(output_dir, "relatorio_sintomas_topicos.txt")
    write_report(df, df_long, report_path, min_symptom_count, n_posts_original)

    logging.info(f"\nTodos os gráficos e o relatório foram salvos em: {output_dir}/")


if __name__ == "__main__":
    input_arg  = sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE
    output_arg = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR
    min_count_arg = int(sys.argv[3]) if len(sys.argv) > 3 else MIN_SYMPTOM_COUNT

    run(input_arg, output_arg, min_count_arg)

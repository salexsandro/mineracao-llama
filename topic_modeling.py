"""
BERTopic - Modelagem de Tópicos para Relatos Reddit
=====================================================
Entrada : relatos_consolidados.jsonl  (22k posts, campo e_relato=true)
Embeddings : all-mpnet-base-v2 (Sentence Transformers)
Redução   : UMAP  (n_neighbors=30)
Clustering: HDBSCAN (min_cluster_size=80)
Representação: c-TF-IDF nativo do BERTopic
"""

import json
import re
import warnings
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import seaborn as sns

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. CONFIGURAÇÕES GERAIS
# ─────────────────────────────────────────────
INPUT_FILE   = "relatos_consolidados.jsonl"
OUTPUT_DIR   = Path("bertopic_output")
EMBED_CACHE  = OUTPUT_DIR / "embeddings.npy"
MODEL_CACHE  = OUTPUT_DIR / "bertopic_model"

OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
MIN_DOC_WORDS = 15          # descarta posts muito curtos
TOP_N_WORDS   = 10          # palavras por tópico no c-TF-IDF

# ─────────────────────────────────────────────
# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO
# ─────────────────────────────────────────────
print("=" * 60)
print("1. CARREGANDO DADOS")
print("=" * 60)

records = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Filtra apenas e_relato=true (por segurança)
        triagem = obj.get("classificacao_triagem", {})
        if not triagem.get("e_relato", False):
            continue

        title = obj.get("title", "") or ""
        body  = obj.get("body",  "") or ""
        text  = (title + " " + body).strip()

        # Remove URLs
        text = re.sub(r"http\S+", "", text)
        # Remove menções reddit
        text = re.sub(r"u/\S+|r/\S+", "", text)
        # Colapsa espaços extras
        text = re.sub(r"\s+", " ", text).strip()

        word_count = len(text.split())
        if word_count < MIN_DOC_WORDS:
            continue

        records.append({
            "post_id"       : obj.get("post_id", ""),
            "subreddit"     : obj.get("subreddit", "unknown"),
            "date"          : obj.get("date", ""),
            "score"         : obj.get("score", 0),
            "comments_count": obj.get("comments_count", 0),
            "text"          : text,
            "word_count"    : word_count,
        })

df = pd.DataFrame(records)
df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
df["year_month"] = df["date"].dt.to_period("M")

docs = df["text"].tolist()

print(f"  Posts carregados : {len(df):,}")
print(f"  Subreddits únicos: {df['subreddit'].nunique()}")
print(f"  Intervalo de datas: {df['date'].min().date()} → {df['date'].max().date()}")
print(f"  Palavras/post  — mín:{df['word_count'].min()}  "
      f"máx:{df['word_count'].max()}  média:{df['word_count'].mean():.0f}")

# ─────────────────────────────────────────────
# 2. EMBEDDINGS  (all-mpnet-base-v2)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. GERANDO EMBEDDINGS  (all-mpnet-base-v2)")
print("=" * 60)

from sentence_transformers import SentenceTransformer

if EMBED_CACHE.exists():
    print(f"  Cache encontrado → carregando {EMBED_CACHE}")
    embeddings = np.load(EMBED_CACHE)
else:
    print("  Nenhum cache encontrado — gerando embeddings…")
    encoder = SentenceTransformer("all-mpnet-base-v2")
    embeddings = encoder.encode(
        docs,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,
    )
    np.save(EMBED_CACHE, embeddings)
    print(f"  Embeddings salvos em {EMBED_CACHE}")

print(f"  Shape: {embeddings.shape}")

# ─────────────────────────────────────────────
# 3. UMAP  (n_neighbors=30)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. REDUÇÃO DE DIMENSIONALIDADE  (UMAP, n_neighbors=30)")
print("=" * 60)

from umap import UMAP

umap_model = UMAP(
    n_neighbors=30,
    n_components=10,      # dimensão interna p/ clustering
    min_dist=0.0,
    metric="cosine",
    random_state=RANDOM_STATE,
    low_memory=False,
)

umap_2d = UMAP(          # projeção 2-D usada apenas para visualização
    n_neighbors=30,
    n_components=2,
    min_dist=0.1,
    metric="cosine",
    random_state=RANDOM_STATE,
)

print("  Ajustando UMAP 10-D para clustering…")
reduced_10d = umap_model.fit_transform(embeddings)

print("  Ajustando UMAP 2-D para visualização…")
reduced_2d = umap_2d.fit_transform(embeddings)

df["umap_x"] = reduced_2d[:, 0]
df["umap_y"] = reduced_2d[:, 1]
print(f"  Projeção 2-D shape: {reduced_2d.shape}")

# ─────────────────────────────────────────────
# 4. CLUSTERING  (HDBSCAN, min_cluster_size=80)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. CLUSTERING  (HDBSCAN, min_cluster_size=80)")
print("=" * 60)

import hdbscan

hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=80,
    min_samples=10,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
)

# ─────────────────────────────────────────────
# 5. BERTOPIC  (c-TF-IDF)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. BERTOPIC  (c-TF-IDF)")
print("=" * 60)

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# Stop-words básicas + termos Reddit genéricos para não poluir os rótulos
EXTRA_STOP = [
    "like", "just", "get", "got", "really", "know", "think", "feel",
    "one", "even", "also", "ve", "im", "it", "use", "using", "used",
    "time", "day", "days", "week", "ago", "reddit", "post", "comment",
    "edit", "update", "deleted", "removed", "amp", "https", "www",
]

vectorizer = CountVectorizer(
    stop_words="english",
    ngram_range=(1, 2),       # unigrams + bigrams
    min_df=5,                 # ignora termos raríssimos
    max_df=0.85,              # ignora termos ubíquos
    vocabulary=None,
)

# Remove extra stop-words da vocabulary depois do fit
# (passadas como token_pattern workaround via subclass não é necessário
#  pois stop_words="english" já cobre muito; adicionamos via fit_transform)

ctfidf = ClassTfidfTransformer(
    reduce_frequent_words=True,
    bm25_weighting=True,
)

if (MODEL_CACHE / "config.json").exists():
    print(f"  Cache de modelo encontrado → carregando {MODEL_CACHE}")
    topic_model = BERTopic.load(str(MODEL_CACHE), embedding_model="all-mpnet-base-v2")
    topics, probs = topic_model.transform(docs, embeddings)
else:
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf,
        top_n_words=TOP_N_WORDS,
        language="english",
        calculate_probabilities=False,
        verbose=True,
        min_topic_size=80,
    )

    topics, probs = topic_model.fit_transform(docs, embeddings)

    topic_model.save(str(MODEL_CACHE), serialization="safetensors",
                     save_ctfidf=True, save_embedding_model="all-mpnet-base-v2")
    print(f"  Modelo salvo em {MODEL_CACHE}")

df["topic"] = topics

# ─────────────────────────────────────────────
# 6. ESTATÍSTICAS DE TÓPICOS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. ESTATÍSTICAS")
print("=" * 60)

topic_info = topic_model.get_topic_info()
n_topics   = len(topic_info[topic_info["Topic"] != -1])
n_noise    = (df["topic"] == -1).sum()

print(f"  Tópicos encontrados  : {n_topics}")
print(f"  Docs classificados   : {len(df) - n_noise:,} ({(len(df)-n_noise)/len(df)*100:.1f}%)")
print(f"  Docs como ruído (-1) : {n_noise:,} ({n_noise/len(df)*100:.1f}%)")
print(f"\n  Top 15 tópicos por tamanho:")
print(topic_info[["Topic","Count","Name"]].head(16).to_string(index=False))

# Enriquece df com nome do tópico
topic_name_map = dict(zip(topic_info["Topic"], topic_info["Name"]))
df["topic_name"] = df["topic"].map(topic_name_map)

# Salva CSV com resultados
out_csv = OUTPUT_DIR / "posts_com_topicos.csv"
df.to_csv(out_csv, index=False)
print(f"\n  Resultados salvos em {out_csv}")

# ─────────────────────────────────────────────
# 7. VISUALIZAÇÕES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. GERANDO VISUALIZAÇÕES")
print("=" * 60)

PALETTE = "tab20"
sns.set_theme(style="whitegrid", font_scale=1.05)

def savefig(name, tight=True):
    path = OUTPUT_DIR / name
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → salvo: {path.name}")

# ── 7.1  Distribuição de tamanho dos tópicos (barras) ──────────────
fig, ax = plt.subplots(figsize=(14, 6))

plot_df = (
    topic_info[topic_info["Topic"] != -1]
    .sort_values("Count", ascending=False)
    .head(30)
)
colors = plt.cm.tab20(np.linspace(0, 1, len(plot_df)))
bars = ax.bar(range(len(plot_df)), plot_df["Count"], color=colors, edgecolor="white", linewidth=0.5)
ax.set_xticks(range(len(plot_df)))
ax.set_xticklabels(
    [f"T{int(t)}" for t in plot_df["Topic"]],
    rotation=45, ha="right", fontsize=9
)
ax.set_xlabel("Tópico")
ax.set_ylabel("Nº de documentos")
ax.set_title("Top 30 tópicos por volume de posts", fontsize=13, fontweight="bold")
for bar, val in zip(bars, plot_df["Count"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
            str(val), ha="center", va="bottom", fontsize=7)
ax.axhline(plot_df["Count"].median(), color="gray", linestyle="--",
           linewidth=1, label=f"Mediana ({plot_df['Count'].median():.0f})")
ax.legend(fontsize=9)
savefig("01_distribuicao_topicos.png")

# ── 7.2  Scatter UMAP 2-D colorido por tópico ──────────────────────
fig, ax = plt.subplots(figsize=(14, 10))

# Ruído em cinza claro
noise = df[df["topic"] == -1]
ax.scatter(noise["umap_x"], noise["umap_y"],
           c="#dddddd", s=4, alpha=0.3, label="Ruído (-1)", rasterized=True)

# Tópicos com cor
topic_ids_sorted = sorted(df[df["topic"] != -1]["topic"].unique())
cmap = plt.cm.get_cmap("tab20", len(topic_ids_sorted))

for i, tid in enumerate(topic_ids_sorted):
    sub = df[df["topic"] == tid]
    ax.scatter(sub["umap_x"], sub["umap_y"],
               c=[cmap(i)], s=6, alpha=0.6,
               label=f"T{tid} ({len(sub)})", rasterized=True)

ax.set_title("Projeção UMAP 2-D — documentos coloridos por tópico",
             fontsize=13, fontweight="bold")
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7,
          markerscale=2, ncol=2, frameon=True)
savefig("02_umap_scatter.png")

# ── 7.3  Heatmap c-TF-IDF: top-5 palavras × top-15 tópicos ────────
top15 = topic_info[topic_info["Topic"] != -1].head(15)["Topic"].tolist()
rows, cols, vals = [], [], []
for tid in top15:
    words = topic_model.get_topic(tid)
    for word, score in words[:5]:
        rows.append(f"T{tid}")
        cols.append(word)
        vals.append(score)

heat_df = (
    pd.DataFrame({"topic": rows, "word": cols, "score": vals})
    .pivot(index="topic", columns="word", values="score")
    .fillna(0)
)

fig, ax = plt.subplots(figsize=(max(16, len(heat_df.columns) * 0.7), 7))
sns.heatmap(
    heat_df,
    cmap="YlOrRd",
    annot=True, fmt=".3f",
    linewidths=0.4, linecolor="white",
    ax=ax, cbar_kws={"label": "Score c-TF-IDF"},
    annot_kws={"size": 7},
)
ax.set_title("Top-5 palavras c-TF-IDF por tópico (15 maiores tópicos)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("Tópico")
plt.xticks(rotation=45, ha="right", fontsize=8)
savefig("03_heatmap_ctfidf.png")

# ── 7.4  Proporção de docs por subreddit de origem ─────────────────
sub_counts = df["subreddit"].value_counts().head(20)
fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.Set2(np.linspace(0, 1, len(sub_counts)))
bars = ax.barh(sub_counts.index[::-1], sub_counts.values[::-1], color=colors[::-1], edgecolor="white")
ax.set_xlabel("Nº de posts")
ax.set_title("Posts por subreddit (top 20)", fontsize=13, fontweight="bold")
for bar, val in zip(bars, sub_counts.values[::-1]):
    ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
            str(val), va="center", fontsize=9)
savefig("04_posts_por_subreddit.png")

# ── 7.5  Evolução temporal: volume de posts por mês ───────────────
monthly = (
    df.dropna(subset=["year_month"])
    .groupby("year_month")
    .size()
    .reset_index(name="count")
)
monthly["ym_str"] = monthly["year_month"].astype(str)

fig, ax = plt.subplots(figsize=(14, 4))
ax.fill_between(monthly["ym_str"], monthly["count"], alpha=0.25, color="#3a86ff")
ax.plot(monthly["ym_str"], monthly["count"], color="#3a86ff", linewidth=1.8)
ax.set_title("Volume de relatos por mês", fontsize=13, fontweight="bold")
ax.set_ylabel("Posts")
ax.set_xlabel("")
step = max(1, len(monthly) // 18)
ax.set_xticks(range(0, len(monthly), step))
ax.set_xticklabels(monthly["ym_str"].iloc[::step], rotation=45, ha="right", fontsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
savefig("05_evolucao_temporal.png")

# ── 7.6  Distribuição do tamanho dos documentos ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df["word_count"], bins=60, color="#8ecae6", edgecolor="white", linewidth=0.4)
axes[0].axvline(df["word_count"].median(), color="#e76f51", linestyle="--",
                label=f"Mediana: {df['word_count'].median():.0f}")
axes[0].set_xlabel("Palavras por post")
axes[0].set_ylabel("Frequência")
axes[0].set_title("Distribuição do tamanho dos posts")
axes[0].legend()

axes[1].hist(np.log1p(df["word_count"]), bins=60, color="#95d5b2", edgecolor="white", linewidth=0.4)
axes[1].set_xlabel("log(1 + palavras)")
axes[1].set_ylabel("Frequência")
axes[1].set_title("Distribuição (escala log)")
savefig("06_distribuicao_tamanho.png")

# ── 7.7  Score médio e mediana por tópico (engajamento) ───────────
top20 = topic_info[topic_info["Topic"] != -1].head(20)["Topic"].tolist()
eng_df = (
    df[df["topic"].isin(top20)]
    .groupby("topic")
    .agg(score_med=("score", "median"),
         score_mean=("score", "mean"),
         comments_med=("comments_count", "median"))
    .reset_index()
    .sort_values("score_med", ascending=False)
)
eng_df["label"] = "T" + eng_df["topic"].astype(str)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = plt.cm.coolwarm(np.linspace(0, 1, len(eng_df)))

# Score mediano
axes[0].barh(eng_df["label"][::-1], eng_df["score_med"][::-1],
             color=colors[::-1], edgecolor="white")
axes[0].set_xlabel("Score mediano (Reddit upvotes)")
axes[0].set_title("Engajamento por tópico\n(score mediano)", fontweight="bold")

# Comentários medianos
eng_df2 = eng_df.sort_values("comments_med", ascending=False)
axes[1].barh(eng_df2["label"][::-1], eng_df2["comments_med"][::-1],
             color=colors[::-1], edgecolor="white")
axes[1].set_xlabel("Comentários medianos")
axes[1].set_title("Engajamento por tópico\n(comentários medianos)", fontweight="bold")

savefig("07_engajamento_por_topico.png")

# ── 7.8  Tópicos mais frequentes por subreddit (stacked bar) ───────
top10_topics = (
    topic_info[topic_info["Topic"] != -1]
    .head(10)["Topic"]
    .tolist()
)
top5_subs = df["subreddit"].value_counts().head(8).index.tolist()

cross = (
    df[df["subreddit"].isin(top5_subs) & df["topic"].isin(top10_topics)]
    .groupby(["subreddit", "topic"])
    .size()
    .reset_index(name="count")
)
cross_pivot = cross.pivot(index="subreddit", columns="topic", values="count").fillna(0)
cross_pivot.columns = [f"T{c}" for c in cross_pivot.columns]

fig, ax = plt.subplots(figsize=(12, 6))
cross_pivot.plot(kind="bar", stacked=True, ax=ax,
                 colormap="tab10", edgecolor="white", linewidth=0.3)
ax.set_title("Distribuição dos top-10 tópicos por subreddit",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Subreddit")
ax.set_ylabel("Nº de posts")
ax.legend(title="Tópico", bbox_to_anchor=(1.01, 1), fontsize=8)
plt.xticks(rotation=30, ha="right")
savefig("08_topicos_por_subreddit.png")

# ── 7.9  Boxplot de score por tópico (top-15) ─────────────────────
top15_ids = topic_info[topic_info["Topic"] != -1].head(15)["Topic"].tolist()
box_df = df[df["topic"].isin(top15_ids)].copy()
box_df["topic_label"] = "T" + box_df["topic"].astype(str)

order = (
    box_df.groupby("topic_label")["score"]
    .median()
    .sort_values(ascending=False)
    .index
    .tolist()
)

fig, ax = plt.subplots(figsize=(14, 5))
sns.boxplot(data=box_df, x="topic_label", y="score", order=order,
            palette="Set3", fliersize=2, linewidth=0.8, ax=ax)
ax.set_yscale("symlog")
ax.set_xlabel("Tópico")
ax.set_ylabel("Score (escala log)")
ax.set_title("Distribuição de score por tópico (top 15)", fontsize=13, fontweight="bold")
plt.xticks(rotation=45, ha="right")
savefig("09_boxplot_score.png")

# ── 7.10  Tabela-resumo: palavras-chave dos tópicos ───────────────
summary_rows = []
for _, row in topic_info[topic_info["Topic"] != -1].head(25).iterrows():
    tid = row["Topic"]
    words = topic_model.get_topic(tid)
    kw = ", ".join([w for w, _ in words[:8]])
    summary_rows.append({
        "Tópico": f"T{tid}",
        "Docs": row["Count"],
        "Palavras-chave (c-TF-IDF)": kw,
    })

summary_df = pd.DataFrame(summary_rows)
out_table = OUTPUT_DIR / "resumo_topicos.csv"
summary_df.to_csv(out_table, index=False)
print(f"  → salvo: resumo_topicos.csv")

fig, ax = plt.subplots(figsize=(16, min(1 + len(summary_df) * 0.45, 14)))
ax.axis("off")
table = ax.table(
    cellText=summary_df.values,
    colLabels=summary_df.columns,
    cellLoc="left",
    loc="center",
    colWidths=[0.07, 0.06, 0.87],
)
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1, 1.6)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2d3047")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#f0f0f0")
    cell.set_edgecolor("#cccccc")
ax.set_title("Resumo dos 25 maiores tópicos — palavras-chave c-TF-IDF",
             fontsize=12, fontweight="bold", pad=10)
savefig("10_tabela_resumo_topicos.png")

# ─────────────────────────────────────────────
# 8. SUMÁRIO FINAL
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("CONCLUÍDO")
print("=" * 60)
print(f"  Tópicos encontrados   : {n_topics}")
print(f"  Docs classificados    : {len(df) - n_noise:,}  ({(len(df)-n_noise)/len(df)*100:.1f}%)")
print(f"  Ruído (-1)            : {n_noise:,}  ({n_noise/len(df)*100:.1f}%)")
print(f"  Arquivos em {OUTPUT_DIR}/")
print("    embeddings.npy          — vetores gerados pelo all-mpnet-base-v2")
print("    bertopic_model/         — modelo BERTopic serializado")
print("    posts_com_topicos.csv   — dataframe completo com coluna 'topic'")
print("    resumo_topicos.csv      — tabela de palavras-chave por tópico")
print("    01_distribuicao_topicos.png")
print("    02_umap_scatter.png")
print("    03_heatmap_ctfidf.png")
print("    04_posts_por_subreddit.png")
print("    05_evolucao_temporal.png")
print("    06_distribuicao_tamanho.png")
print("    07_engajamento_por_topico.png")
print("    08_topicos_por_subreddit.png")
print("    09_boxplot_score.png")
print("    10_tabela_resumo_topicos.png")
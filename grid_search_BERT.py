"""
Grid search over UMAP_N_COMPONENTS × UMAP_N_NEIGHBORS for BERTopic.

For each (n_components, n_neighbors) combination:
  1. UMAP reduction
  2. Elbow + Silhouette to pick optimal K
  3. KMeans clustering
  4. BERTopic fit
  5. Silhouette, BetaCV, C_v coherence

Outputs per combination:
  grid_results/{nc}c_{nn}n/topics_output.json
  grid_results/{nc}c_{nn}n/posts_with_topics.jsonl

Final summary:
  grid_results/summary.txt
"""

import json
import os
import re
import warnings
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer

from scipy.special import softmax
from umap import UMAP
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

warnings.filterwarnings("ignore")

for pkg in ["stopwords", "punkt", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE    = "reddit_relevant_only_full.jsonl"
OUTPUT_DIR    = "grid_results"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Grid to search
N_COMPONENTS_LIST = [5, 10, 15, 20]
N_NEIGHBORS_LIST  = [10, 15, 20, 25, 30, 40, 50]

# K search range (same for all combinations)
K_MIN, K_MAX = 2, 20

# Fixed UMAP settings
UMAP_MIN_DIST = 0.0    # keep 0.0 for KMeans — tighter clusters

# Minimum tokens per document
MIN_TOKENS = 5

# C_v coherence top-N words per topic
TOP_N_COHERENCE = 10

# ── Custom stopwords ──────────────────────────────────────────────────────────
EXTRA_STOPWORDS = {
    "tiktok", "youtube", "instagram", "reels", "shorts", "snapchat", "vine",
    "reddit", "post", "comment", "upvote", "downvote", "subreddit", "edit",
    "deleted", "removed", "mod", "karma",
    "like", "just", "really", "actually", "also", "even", "still", "much",
    "get", "got", "go", "going", "know", "think", "feel", "felt", "make",
    "made", "want", "wanted", "need", "use", "using", "used", "one", "time",
    "day", "would", "could", "people", "thing", "things", "lot", "way",
    "said", "say", "see", "saw", "come", "came", "back", "around", "since",
    "every", "always", "never", "already", "though", "something", "someone",
    "anyone", "everyone", "nothing", "anything", "everything",
    "lol", "lmao", "idk", "imo", "tbh", "ngl", "omg", "wtf", "bc", "cuz",
    "tho", "rn", "irl", "aka", "btw", "fyi", "dms", "dm",
}
KEEP_WORDS = {"no", "not", "without", "cannot", "can't", "don't", "won't"}


# =============================================================================
# HELPERS
# =============================================================================

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_filter(text: str, stopwords_set: set) -> str:
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords_set and len(t) > 2]
    return " ".join(tokens)


def compute_betacv(emb, labels):
    unique = [l for l in set(labels) if l != -1]
    if len(unique) < 2:
        return float("nan")
    centroids = {l: emb[np.array(labels) == l].mean(axis=0) for l in unique}
    intra = [np.linalg.norm(emb[np.array(labels) == l] - centroids[l], axis=1).mean()
             for l in unique]
    cvec = list(centroids.values())
    inter = [np.linalg.norm(cvec[i] - cvec[j])
             for i in range(len(cvec)) for j in range(i + 1, len(cvec))]
    return float(np.mean(intra) / np.mean(inter)) if inter else float("nan")


def pick_optimal_k(inertias, silhouettes, k_range):
    """Silhouette peak; tie-break with elbow (second derivative of inertia)."""
    best_k_sil = list(k_range)[int(np.argmax(silhouettes))]
    d2 = np.diff(np.diff(np.array(inertias)))
    best_k_elbow = list(k_range)[int(np.argmax(np.abs(d2))) + 1]
    return best_k_sil, best_k_elbow


def compute_coherence(cleaned, topic_model, valid_topic_ids):
    tokenized  = [t.split() for t in cleaned]
    dictionary = Dictionary(tokenized)
    dictionary.filter_extremes(no_below=2, no_above=0.85)
    topic_words_list = []
    for tid in valid_topic_ids:
        ws = topic_model.get_topic(tid)
        if ws:
            words = [w for w, _ in ws[:TOP_N_COHERENCE] if w in dictionary.token2id]
            if len(words) >= 2:
                topic_words_list.append(words)
    if not topic_words_list:
        return float("nan"), []
    cm = CoherenceModel(
        topics=topic_words_list,
        texts=tokenized,
        dictionary=dictionary,
        coherence="c_v",
        processes=1,
    )
    return cm.get_coherence(), cm.get_coherence_per_topic()


def get_topic_label(topic_model, tid, n=5):
    ws = topic_model.get_topic(tid)
    if not ws:
        return "unknown"
    return "_".join(w for w, _ in ws[:n])


def run_combination(
    nc, nn,
    embeddings, cleaned, records, raw_texts,
    stopwords_set, embedder,
    out_dir
):
    """
    Run the full pipeline for one (n_components, n_neighbors) pair.
    Returns a dict with all evaluation metrics.
    """
    os.makedirs(out_dir, exist_ok=True)
    tag = f"nc={nc}, nn={nn}"
    print(f"\n{'='*60}")
    print(f"  Running: {tag}")
    print(f"{'='*60}")

    # ── UMAP ──────────────────────────────────────────────────────────────────
    umap_model = UMAP(
        n_components=nc,
        n_neighbors=nn,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=42,
    )
    reduced      = umap_model.fit_transform(embeddings)
    reduced_norm = normalize(reduced)

    # ── Elbow + Silhouette → optimal K ────────────────────────────────────────
    k_range    = range(K_MIN, K_MAX + 1)
    inertias, silhouettes = [], []

    for k in k_range:
        km  = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(reduced_norm)
        inertias.append(km.inertia_)
        sil = silhouette_score(
            reduced_norm, lbl,
            sample_size=min(3000, len(lbl))
        )
        silhouettes.append(sil)

    best_k_sil, best_k_elbow = pick_optimal_k(inertias, silhouettes, k_range)
    optimal_k = best_k_sil
    print(f"  Elbow K={best_k_elbow}  |  Silhouette K={best_k_sil}  →  Using K={optimal_k}")

    # ── Elbow / Silhouette plot ────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(list(k_range), inertias, "bo-", lw=2, ms=6)
    ax1.axvline(best_k_elbow, color="red",   ls="--", label=f"Elbow K={best_k_elbow}")
    ax1.axvline(optimal_k,    color="green", ls="--", label=f"Chosen K={optimal_k}")
    ax1.set_xlabel("K"); ax1.set_ylabel("Inertia (WCSS)")
    ax1.set_title(f"Elbow ({tag})"); ax1.legend(); ax1.grid(alpha=.3)
    ax2.plot(list(k_range), silhouettes, "rs-", lw=2, ms=6)
    ax2.axvline(best_k_sil, color="red",   ls="--", label=f"Sil. peak K={best_k_sil}")
    ax2.axvline(optimal_k,  color="green", ls="--", label=f"Chosen K={optimal_k}")
    ax2.set_xlabel("K"); ax2.set_ylabel("Silhouette Score")
    ax2.set_title(f"Silhouette ({tag})"); ax2.legend(); ax2.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "elbow_silhouette.png"), dpi=150)
    plt.close()

    # ── Final KMeans ──────────────────────────────────────────────────────────
    km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
    km_final.fit(reduced_norm)

    # ── BERTopic ──────────────────────────────────────────────────────────────
    umap_bertopic = UMAP(
        n_components=nc,
        n_neighbors=nn,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=42,
    )
    vectorizer = CountVectorizer(
        stop_words=list(stopwords_set),
        min_df=2,
        max_df=0.85,
        ngram_range=(1, 2),
    )
    ctfidf = ClassTfidfTransformer(reduce_frequent_words=True)
    topic_model = BERTopic(
        embedding_model=embedder,
        umap_model=umap_bertopic,
        hdbscan_model=km_final,
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf,
        nr_topics=optimal_k,
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(cleaned, embeddings=embeddings)
    topic_info = topic_model.get_topic_info()

    # ── Confidence ────────────────────────────────────────────────────────────
    centroids  = km_final.cluster_centers_
    diffs      = reduced_norm[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    distances  = np.linalg.norm(diffs, axis=2)
    soft       = softmax(-distances, axis=1)
    assigned   = np.array(topics)
    # Guard against out-of-range topic indices (BERTopic may remap)
    safe_assigned = np.clip(assigned, 0, soft.shape[1] - 1)
    confidences   = soft[np.arange(len(safe_assigned)), safe_assigned].tolist()
    all_probs     = soft.tolist()

    # ── Save per-post JSONL ───────────────────────────────────────────────────
    jsonl_path = os.path.join(out_dir, "posts_with_topics.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i, record in enumerate(records):
            topic_id = int(topics[i])
            probs_i  = all_probs[i]
            top3     = sorted(enumerate(probs_i), key=lambda x: -x[1])
            alts     = [{"topic_id": int(tid), "prob": round(p, 4)}
                        for tid, p in top3 if int(tid) != topic_id][:2]
            out = {
                "post_id":            record.get("post_id"),
                "title":              record.get("title"),
                "body":               record.get("body"),
                "topic_id":           topic_id,
                "topic_label":        get_topic_label(topic_model, topic_id),
                "confidence":         round(confidences[i], 4),
                "alternative_topics": alts,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    # ── Evaluation metrics ────────────────────────────────────────────────────
    valid_mask = np.array(topics) != -1
    if valid_mask.sum() > 1 and len(set(np.array(topics)[valid_mask])) > 1:
        sil_final = silhouette_score(
            reduced_norm[valid_mask],
            np.array(topics)[valid_mask],
            sample_size=min(5000, int(valid_mask.sum())),
        )
    else:
        sil_final = float("nan")

    betacv = compute_betacv(reduced_norm, topics)

    valid_topic_ids = [tid for tid in topic_model.get_topics() if tid != -1]
    coherence_cv, per_topic_coh = compute_coherence(cleaned, topic_model, valid_topic_ids)

    print(f"  Silhouette={sil_final:.4f}  BetaCV={betacv:.4f}  C_v={coherence_cv:.4f}")

    # ── Save topics JSON ──────────────────────────────────────────────────────
    results = {
        "config": {
            "n_components": nc,
            "n_neighbors":  nn,
            "optimal_k":    optimal_k,
            "best_k_elbow": best_k_elbow,
            "best_k_silhouette": best_k_sil,
        },
        "evaluation": {
            "silhouette_score": round(sil_final, 4) if not np.isnan(sil_final) else None,
            "betacv":           round(betacv, 4)    if not np.isnan(betacv)    else None,
            "mean_coherence_cv": round(coherence_cv, 4) if not np.isnan(coherence_cv) else None,
        },
        "topics": [],
    }
    for i, tid in enumerate(valid_topic_ids):
        ws       = topic_model.get_topic(tid)
        tw       = [{"word": w, "score": round(s, 5)} for w, s in ws[:15]] if ws else []
        coh_val  = per_topic_coh[i] if i < len(per_topic_coh) else None
        size_row = topic_info.loc[topic_info["Topic"] == tid, "Count"]
        size     = int(size_row.values[0]) if len(size_row) > 0 else 0
        results["topics"].append({
            "topic_id":     int(tid),
            "size":         size,
            "coherence_cv": round(coh_val, 4) if isinstance(coh_val, float) else None,
            "top_words":    tw,
        })

    with open(os.path.join(out_dir, "topics_output.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return {
        "n_components":    nc,
        "n_neighbors":     nn,
        "optimal_k":       optimal_k,
        "best_k_elbow":    best_k_elbow,
        "best_k_sil":      best_k_sil,
        "silhouette":      sil_final,
        "betacv":          betacv,
        "coherence_cv":    coherence_cv,
        "topics":          results["topics"],
        "out_dir":         out_dir,
    }


def write_summary(all_results, output_path, n_docs):
    """Write a human-readable summary TXT with ranking and best config detail."""

    # ── Rank by composite score ───────────────────────────────────────────────
    # Silhouette ↑, BetaCV ↓ (invert for ranking), C_v ↑
    # Normalize each metric to [0,1] then compute weighted average
    def safe(v):
        return v if (v is not None and not np.isnan(v)) else None

    sil_vals  = [safe(r["silhouette"])   for r in all_results]
    bcv_vals  = [safe(r["betacv"])       for r in all_results]
    coh_vals  = [safe(r["coherence_cv"]) for r in all_results]

    def norm_minmax(vals, invert=False):
        valid = [v for v in vals if v is not None]
        if not valid or max(valid) == min(valid):
            return [0.5 if v is not None else None for v in vals]
        lo, hi = min(valid), max(valid)
        normed = [(v - lo) / (hi - lo) if v is not None else None for v in vals]
        if invert:
            normed = [(1 - v) if v is not None else None for v in normed]
        return normed

    sil_n = norm_minmax(sil_vals, invert=False)   # higher = better
    bcv_n = norm_minmax(bcv_vals, invert=True)    # lower = better → invert
    coh_n = norm_minmax(coh_vals, invert=False)   # higher = better

    # Weights: silhouette 40%, betacv 30%, coherence 30%
    W_SIL, W_BCV, W_COH = 0.40, 0.30, 0.30

    scores = []
    for s, b, c in zip(sil_n, bcv_n, coh_n):
        parts = []
        if s is not None: parts.append((W_SIL, s))
        if b is not None: parts.append((W_BCV, b))
        if c is not None: parts.append((W_COH, c))
        if parts:
            total_w = sum(w for w, _ in parts)
            score   = sum(w * v for w, v in parts) / total_w
        else:
            score = 0.0
        scores.append(score)

    ranked = sorted(
        zip(scores, all_results),
        key=lambda x: x[0],
        reverse=True
    )

    best_score, best = ranked[0]

    lines = []
    lines.append("=" * 70)
    lines.append("  UMAP GRID SEARCH — SUMMARY")
    lines.append("=" * 70)
    lines.append(f"  Documents analysed : {n_docs}")
    lines.append(f"  Embedding model    : {EMBEDDING_MODEL}")
    lines.append(f"  N_COMPONENTS tested: {N_COMPONENTS_LIST}")
    lines.append(f"  N_NEIGHBORS tested : {N_NEIGHBORS_LIST}")
    lines.append(f"  K range            : [{K_MIN}, {K_MAX}]")
    lines.append(f"  Total combinations : {len(all_results)}")
    lines.append("")

    # ── Ranking table ─────────────────────────────────────────────────────────
    lines.append("─" * 70)
    lines.append("  RANKING  (composite score = 40% Sil + 30% BetaCV↓ + 30% C_v)")
    lines.append("─" * 70)
    header = f"  {'Rank':>4}  {'nc':>4}  {'nn':>4}  {'K':>3}  {'Silhouette':>10}  {'BetaCV':>8}  {'C_v':>8}  {'Score':>8}"
    lines.append(header)
    lines.append("  " + "-" * 66)

    for rank, (sc, r) in enumerate(ranked, 1):
        sil_str = f"{r['silhouette']:.4f}" if r['silhouette'] is not None and not np.isnan(r['silhouette']) else "   N/A  "
        bcv_str = f"{r['betacv']:.4f}"     if r['betacv']    is not None and not np.isnan(r['betacv'])    else "   N/A  "
        coh_str = f"{r['coherence_cv']:.4f}" if r['coherence_cv'] is not None and not np.isnan(r['coherence_cv']) else "   N/A  "
        marker  = "  ← BEST" if rank == 1 else ""
        lines.append(
            f"  {rank:>4}  {r['n_components']:>4}  {r['n_neighbors']:>4}  "
            f"{r['optimal_k']:>3}  {sil_str:>10}  {bcv_str:>8}  {coh_str:>8}  "
            f"{sc:>8.4f}{marker}"
        )

    lines.append("")

    # ── Best configuration detail ─────────────────────────────────────────────
    lines.append("=" * 70)
    lines.append("  BEST CONFIGURATION DETAIL")
    lines.append("=" * 70)
    lines.append(f"  n_components : {best['n_components']}")
    lines.append(f"  n_neighbors  : {best['n_neighbors']}")
    lines.append(f"  Optimal K    : {best['optimal_k']}  "
                 f"(elbow={best['best_k_elbow']}, silhouette peak={best['best_k_sil']})")
    lines.append("")
    lines.append("  Evaluation metrics:")

    def fmt(v, decimals=4):
        return f"{v:.{decimals}f}" if (v is not None and not np.isnan(v)) else "N/A"

    lines.append(f"    Silhouette Score : {fmt(best['silhouette'])}  (↑ better, range [-1,1])")
    lines.append(f"    BetaCV           : {fmt(best['betacv'])}  (↓ better)")
    lines.append(f"    Mean C_v         : {fmt(best['coherence_cv'])}  (↑ better, range [0,1])")
    lines.append(f"    Composite Score  : {best_score:.4f}")
    lines.append("")
    lines.append(f"  Output files: {best['out_dir']}/")
    lines.append("")

    # ── Topic breakdown for best config ───────────────────────────────────────
    lines.append("─" * 70)
    lines.append("  TOPICS — BEST CONFIGURATION")
    lines.append("─" * 70)
    for t in sorted(best["topics"], key=lambda x: -x["size"]):
        top_words = [tw["word"] for tw in t["top_words"][:8]]
        coh_str   = f"{t['coherence_cv']:.4f}" if t["coherence_cv"] is not None else "N/A"
        lines.append(f"\n  Topic {t['topic_id']:>2}  (n={t['size']:>5}, C_v={coh_str})")
        lines.append(f"    Words: {', '.join(top_words)}")

    lines.append("")

    # ── Analysis notes ────────────────────────────────────────────────────────
    lines.append("─" * 70)
    lines.append("  ANALYSIS NOTES")
    lines.append("─" * 70)

    # Silhouette interpretation
    best_sil = best["silhouette"]
    if best_sil is not None and not np.isnan(best_sil):
        if best_sil >= 0.5:
            sil_interp = "strong cluster separation"
        elif best_sil >= 0.25:
            sil_interp = "moderate cluster separation — acceptable for topic modelling"
        else:
            sil_interp = "weak cluster separation — consider fewer topics or different preprocessing"
        lines.append(f"  Silhouette {best_sil:.4f}: {sil_interp}.")

    # BetaCV interpretation
    best_bcv = best["betacv"]
    if best_bcv is not None and not np.isnan(best_bcv):
        if best_bcv < 0.3:
            bcv_interp = "tight, well-separated clusters"
        elif best_bcv < 0.6:
            bcv_interp = "moderate compactness"
        else:
            bcv_interp = "loose clusters — overlap between topics likely"
        lines.append(f"  BetaCV {best_bcv:.4f}: {bcv_interp}.")

    # C_v interpretation
    best_coh = best["coherence_cv"]
    if best_coh is not None and not np.isnan(best_coh):
        if best_coh >= 0.6:
            coh_interp = "high coherence — topics are semantically meaningful"
        elif best_coh >= 0.4:
            coh_interp = "moderate coherence — topics interpretable but with some noise"
        else:
            coh_interp = "low coherence — topics may be too broad or noisy"
        lines.append(f"  C_v {best_coh:.4f}: {coh_interp}.")

    # Compare top-3
    lines.append("")
    lines.append("  Top 3 configurations for reference:")
    for rank, (sc, r) in enumerate(ranked[:3], 1):
        lines.append(
            f"    {rank}. nc={r['n_components']}, nn={r['n_neighbors']}, K={r['optimal_k']}"
            f"  →  Sil={fmt(r['silhouette'])}, BCV={fmt(r['betacv'])}, C_v={fmt(r['coherence_cv'])}"
            f"  (score={sc:.4f})"
        )

    lines.append("")
    lines.append("=" * 70)
    lines.append("  END OF SUMMARY")
    lines.append("=" * 70)

    text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(text)
    return text


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Load & preprocess (done once) ──────────────────────────────────────
    records = load_jsonl(INPUT_FILE)
    print(f"Loaded {len(records)} posts.")

    base_sw   = set(stopwords.words("english"))
    STOPWORDS = (base_sw | EXTRA_STOPWORDS) - KEEP_WORDS

    raw_texts = [f"{r.get('title','') or ''}. {r.get('body','') or ''}" for r in records]
    cleaned   = [tokenize_and_filter(clean_text(t), STOPWORDS) for t in raw_texts]

    valid     = [i for i, t in enumerate(cleaned) if len(t.split()) >= MIN_TOKENS]
    cleaned   = [cleaned[i]   for i in valid]
    raw_texts = [raw_texts[i] for i in valid]
    records   = [records[i]   for i in valid]
    print(f"After filtering short posts: {len(cleaned)} documents.")

    # ── 2. Embed (done once — most expensive step) ────────────────────────────
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nEmbedding on {device.upper()} with '{EMBEDDING_MODEL}' ...")
    embedder   = SentenceTransformer(EMBEDDING_MODEL, device=device)
    embeddings = embedder.encode(cleaned, show_progress_bar=True, batch_size=64)
    print(f"Embedding shape: {embeddings.shape}")

    # ── 3. Grid search ────────────────────────────────────────────────────────
    combinations = list(itertools.product(N_COMPONENTS_LIST, N_NEIGHBORS_LIST))
    total        = len(combinations)
    print(f"\nGrid search: {len(N_COMPONENTS_LIST)} × {len(N_NEIGHBORS_LIST)} = {total} combinations")

    all_results = []

    for idx, (nc, nn) in enumerate(combinations, 1):
        print(f"\n[{idx}/{total}] n_components={nc}, n_neighbors={nn}")

        # Skip if nn >= number of documents (UMAP constraint)
        if nn >= len(cleaned):
            print(f"  Skipping: n_neighbors ({nn}) >= n_docs ({len(cleaned)})")
            continue

        combo_dir = os.path.join(OUTPUT_DIR, f"{nc}c_{nn}n")

        try:
            result = run_combination(
                nc=nc, nn=nn,
                embeddings=embeddings,
                cleaned=cleaned,
                records=records,
                raw_texts=raw_texts,
                stopwords_set=STOPWORDS,
                embedder=embedder,
                out_dir=combo_dir,
            )
            all_results.append(result)

        except Exception as e:
            print(f"  ERROR for nc={nc}, nn={nn}: {e}")
            all_results.append({
                "n_components": nc, "n_neighbors": nn,
                "optimal_k": None, "best_k_elbow": None, "best_k_sil": None,
                "silhouette": None, "betacv": None, "coherence_cv": None,
                "topics": [], "out_dir": combo_dir,
            })

    # ── 4. Summary ────────────────────────────────────────────────────────────
    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    write_summary(all_results, summary_path, n_docs=len(cleaned))
    print(f"\nSummary saved → {summary_path}")
    print("\nDone ✓")
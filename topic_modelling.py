"""
BERTopic modelling on Reddit short-video / mental-health posts.
Pipeline:
  1. Load & preprocess text (title + body)
  2. Embed with SentenceTransformer (GPU if available)
  3. UMAP dimensionality reduction (makes KMeans / elbow meaningful)
  4. Elbow method (inertia) + Silhouette to pick optimal K
  5. KMeans clustering on UMAP space
  6. BERTopic with the pre-fitted KMeans model
  7. Evaluation: Silhouette, BetaCV, Topic Coherence (C_v)

Windows note: ALL logic is inside  if __name__ == "__main__"  so that
Gensim's multiprocessing workers don't crash on spawn.
"""

import json
import re
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")                   # non-interactive backend (safe on all OS)
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
# 0. CONFIGURATION  ← edit here
# =============================================================================

INPUT_FILE    = "reddit_relevant_only.jsonl"
OUTPUT_PLOTS  = "elbow_silhouette.png"
OUTPUT_TOPICS = "topics_output.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# K search range
K_MIN, K_MAX = 2, 20

# UMAP settings — reduce to this many dimensions before KMeans
# 5-15 is typical; lower = more aggressive compression
UMAP_N_COMPONENTS = 10
UMAP_N_NEIGHBORS  = 15     # larger = more global structure preserved
UMAP_MIN_DIST     = 0.0    # 0.0 tightens clusters (good for KMeans)

# ── Custom stopwords ──────────────────────────────────────────────────────────
EXTRA_STOPWORDS = {
    # platform names — remove these if you want platform as a signal
    "tiktok", "youtube", "instagram", "reels", "shorts", "snapchat", "vine",
    # generic Reddit noise
    "reddit", "post", "comment", "upvote", "downvote", "subreddit", "edit",
    "deleted", "removed", "mod", "karma",
    # high-frequency low-signal words
    "like", "just", "really", "actually", "also", "even", "still", "much",
    "get", "got", "go", "going", "know", "think", "feel", "felt", "make",
    "made", "want", "wanted", "need", "use", "using", "used", "one", "time",
    "day", "would", "could", "people", "thing", "things", "lot", "way",
    "said", "say", "see", "saw", "come", "came", "back", "around", "since",
    "every", "always", "never", "already", "though", "something", "someone",
    "anyone", "everyone", "nothing", "anything", "everything",
    # internet slang noise
    "lol", "lmao", "idk", "imo", "tbh", "ngl", "omg", "wtf", "bc", "cuz",
    "tho", "rn", "irl", "aka", "btw", "fyi", "dms", "dm",
}

# Words to KEEP even if NLTK would remove them
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
    """
    BetaCV = mean intra-cluster distance / mean inter-centroid distance.
    Lower is better (tight, well-separated clusters).
    """
    unique = [l for l in set(labels) if l != -1]
    if len(unique) < 2:
        return float("nan")
    centroids = {l: emb[np.array(labels) == l].mean(axis=0) for l in unique}
    intra = [np.linalg.norm(emb[np.array(labels) == l] - centroids[l], axis=1).mean()
             for l in unique]
    cvec = list(centroids.values())
    inter = [np.linalg.norm(cvec[i] - cvec[j])
             for i in range(len(cvec)) for j in range(i+1, len(cvec))]
    return float(np.mean(intra) / np.mean(inter)) if inter else float("nan")


# =============================================================================
# MAIN — must be guarded for Windows multiprocessing
# =============================================================================

if __name__ == "__main__":

    # ── 1. Load ───────────────────────────────────────────────────────────────
    records = load_jsonl(INPUT_FILE)
    print(f"Loaded {len(records)} posts.")

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    base_sw  = set(stopwords.words("english"))
    STOPWORDS = (base_sw | EXTRA_STOPWORDS) - KEEP_WORDS

    raw_texts = [f"{r.get('title','') or ''}. {r.get('body','') or ''}" for r in records]
    cleaned   = [tokenize_and_filter(clean_text(t), STOPWORDS) for t in raw_texts]

    MIN_TOKENS = 5
    valid = [i for i, t in enumerate(cleaned) if len(t.split()) >= MIN_TOKENS]
    cleaned   = [cleaned[i]   for i in valid]
    raw_texts = [raw_texts[i] for i in valid]
    records   = [records[i]   for i in valid]
    print(f"After filtering short posts: {len(cleaned)} documents.")

    # ── 3. Embed ──────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nEmbedding on {device.upper()} with '{EMBEDDING_MODEL}' ...")
    embedder   = SentenceTransformer(EMBEDDING_MODEL, device=device)
    embeddings = embedder.encode(cleaned, show_progress_bar=True, batch_size=64)
    print(f"Embedding shape: {embeddings.shape}")

    # ── 4. UMAP reduction ─────────────────────────────────────────────────────
    # Why: KMeans + silhouette are unreliable in 384-dim space (curse of
    # dimensionality). UMAP compresses to a lower-dim space where distances
    # are meaningful, giving better elbow curves and cluster separation.
    print(f"\nReducing to {UMAP_N_COMPONENTS}D with UMAP ...")
    umap_model = UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=42,
    )
    reduced = umap_model.fit_transform(embeddings)
    reduced_norm = normalize(reduced)
    print(f"Reduced shape:   {reduced.shape}")

    # ── 5. Elbow + Silhouette on UMAP space ───────────────────────────────────
    print(f"\nSearching K in [{K_MIN}, {K_MAX}] on UMAP-reduced embeddings ...")
    inertias, silhouettes = [], []
    K_range = range(K_MIN, K_MAX + 1)

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(reduced_norm)
        inertias.append(km.inertia_)
        sil = silhouette_score(reduced_norm, lbl, sample_size=min(3000, len(lbl)))
        silhouettes.append(sil)
        print(f"  K={k:2d}  inertia={km.inertia_:.2f}  silhouette={sil:.4f}")

    # Elbow via second derivative
    d2 = np.diff(np.diff(np.array(inertias)))
    best_k_elbow = list(K_range)[int(np.argmax(np.abs(d2))) + 1]
    best_k_sil   = list(K_range)[int(np.argmax(silhouettes))]

    print(f"\nElbow method    → best K = {best_k_elbow}")
    print(f"Silhouette peak → best K = {best_k_sil}")

    # ── Decision: prefer silhouette peak; tie-break with elbow ────────────────
    # You can also override manually: OPTIMAL_K = 7
    OPTIMAL_K = best_k_sil
    print(f"Using K = {OPTIMAL_K}  (set OPTIMAL_K manually above to override)\n")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(list(K_range), inertias, "bo-", lw=2, ms=6)
    ax1.axvline(best_k_elbow, color="red",   ls="--", label=f"Elbow K={best_k_elbow}")
    ax1.axvline(OPTIMAL_K,    color="green", ls="--", label=f"Chosen K={OPTIMAL_K}")
    ax1.set_xlabel("K"); ax1.set_ylabel("Inertia (WCSS)")
    ax1.set_title("Elbow Method (UMAP space)"); ax1.legend(); ax1.grid(alpha=.3)

    ax2.plot(list(K_range), silhouettes, "rs-", lw=2, ms=6)
    ax2.axvline(best_k_sil, color="red",   ls="--", label=f"Sil. peak K={best_k_sil}")
    ax2.axvline(OPTIMAL_K,  color="green", ls="--", label=f"Chosen K={OPTIMAL_K}")
    ax2.set_xlabel("K"); ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score vs K (UMAP space)"); ax2.legend(); ax2.grid(alpha=.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOTS, dpi=150)
    plt.close()
    print(f"Elbow/Silhouette plot saved → {OUTPUT_PLOTS}")

    # ── 6. Final KMeans ───────────────────────────────────────────────────────
    km_final = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=20)
    km_final.fit(reduced_norm)

    # ── 7. BERTopic ───────────────────────────────────────────────────────────
    print("\nFitting BERTopic ...")

    # Use a fresh UMAP for BERTopic (it uses 2D internally for visualisation)
    umap_bertopic = UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=42,
    )

    vectorizer = CountVectorizer(
        stop_words=list(STOPWORDS),
        min_df=2,
        max_df=0.85,
        ngram_range=(1, 2),
    )
    ctfidf = ClassTfidfTransformer(reduce_frequent_words=True)

    topic_model = BERTopic(
        embedding_model=embedder,
        umap_model=umap_bertopic,
        hdbscan_model=km_final,       # KMeans instead of HDBSCAN
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf,
        nr_topics=OPTIMAL_K,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(cleaned, embeddings=embeddings)

    topic_info = topic_model.get_topic_info()

    def compute_kmeans_confidence(reduced_norm, km_model):
        """
        For each point, compute softmax over negative distances to all centroids.
        The probability assigned to the closest centroid = confidence.
        """
        centroids   = km_model.cluster_centers_          # shape (K, n_components)
        # Euclidean distance from every point to every centroid
        # shape: (n_docs, K)
        diffs       = reduced_norm[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        distances   = np.linalg.norm(diffs, axis=2)
    
        # Softmax over *negative* distances → high prob = close to centroid
        neg_dists   = -distances
        soft        = softmax(neg_dists, axis=1)          # shape (n_docs, K)
    
        # Confidence = probability of the assigned cluster
        assigned    = np.array(topics)
        confidence  = soft[np.arange(len(assigned)), assigned]
        return confidence.tolist(), soft.tolist()
    
    confidences, all_probs = compute_kmeans_confidence(reduced_norm, km_final)
    
    # ── Save per-post assignments ─────────────────────────────────────────────────
    OUTPUT_ASSIGNMENTS = "posts_with_topics.jsonl"
    
    def get_topic_label(tid, n=5):
        ws = topic_model.get_topic(tid)
        if not ws:
            return "unknown"
        return "_".join(w for w, _ in ws[:n])
    
    with open(OUTPUT_ASSIGNMENTS, "w", encoding="utf-8") as f:
        for i, record in enumerate(records):
            topic_id = int(topics[i])
            # Top 3 alternative topics by probability (excluding assigned)
            probs_i  = all_probs[i]
            top3     = sorted(enumerate(probs_i), key=lambda x: -x[1])
            alts     = [{"topic_id": int(tid), "prob": round(p, 4)}
                        for tid, p in top3 if int(tid) != topic_id][:2]
    
            out = {
                "post_id":            record.get("post_id"),
                "title":              record.get("title"),
                "body":               record.get("body"),
                "topic_id":           topic_id,
                "topic_label":        get_topic_label(topic_id),
                "confidence":         round(confidences[i], 4),
                "alternative_topics": alts,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    
    print(f"Per-post assignments saved → {OUTPUT_ASSIGNMENTS}")
    
    # ── Summary ───────────────────────────────────────────────────────────────────
    from collections import Counter
    topic_counts = Counter(topics)
    
    print("\n── Posts per topic ──")
    for tid in sorted(topic_counts):
        label       = get_topic_label(tid)
        topic_confs = [confidences[i] for i, t in enumerate(topics) if t == tid]
        avg_conf    = sum(topic_confs) / len(topic_confs)
        print(f"  Topic {tid:2d}  ({topic_counts[tid]:4d} posts, avg conf={avg_conf:.3f})  [{label}]")
    
    # Flag low-confidence posts (border cases worth reviewing)
    LOW_CONF_THRESHOLD = 0.4
    low_conf = [(i, topics[i], confidences[i]) for i in range(len(topics))
                if confidences[i] < LOW_CONF_THRESHOLD]
    print(f"\n  {len(low_conf)} posts below confidence threshold ({LOW_CONF_THRESHOLD}) — border cases")


    # ── 8. Evaluation ─────────────────────────────────────────────────────────

    # 8a. Silhouette on UMAP-reduced space
    valid_mask = np.array(topics) != -1
    if valid_mask.sum() > 1 and len(set(np.array(topics)[valid_mask])) > 1:
        sil_final = silhouette_score(
            reduced_norm[valid_mask],
            np.array(topics)[valid_mask],
            sample_size=min(5000, int(valid_mask.sum())),
        )
    else:
        sil_final = float("nan")
    print(f"\nSilhouette Score (UMAP space): {sil_final:.4f}")

    # 8b. BetaCV
    betacv = compute_betacv(reduced_norm, topics)
    print(f"BetaCV (lower = better):       {betacv:.4f}")

    # 8c. C_v coherence — runs in main process (processes=1) to avoid Windows crash
    print("\nComputing C_v coherence (single-process mode for Windows compatibility) ...")
    tokenized = [t.split() for t in cleaned]
    dictionary = Dictionary(tokenized)
    dictionary.filter_extremes(no_below=2, no_above=0.85)

    TOP_N = 10
    valid_topic_ids = [tid for tid in topic_model.get_topics() if tid != -1]
    topic_words_list = []
    for tid in valid_topic_ids:
        ws = topic_model.get_topic(tid)
        if ws:
            words = [w for w, _ in ws[:TOP_N] if w in dictionary.token2id]
            if len(words) >= 2:
                topic_words_list.append(words)

    if topic_words_list:
        cm = CoherenceModel(
            topics=topic_words_list,
            texts=tokenized,
            dictionary=dictionary,
            coherence="c_v",
            processes=1,            # ← single process: fixes Windows spawn crash
        )
        coherence_cv  = cm.get_coherence()
        per_topic_coh = cm.get_coherence_per_topic()
    else:
        coherence_cv, per_topic_coh = float("nan"), []

    print(f"Mean C_v Coherence: {coherence_cv:.4f}")

    # ── 9. Print full results ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  TOPIC MODELLING RESULTS")
    print("="*60)
    print(f"  Documents:          {len(cleaned)}")
    print(f"  Optimal K:          {OPTIMAL_K}")
    print(f"  Silhouette Score:   {sil_final:.4f}  (↑ better, range [-1,1])")
    print(f"  BetaCV:             {betacv:.4f}    (↓ better)")
    print(f"  Mean C_v Coherence: {coherence_cv:.4f}  (↑ better, range [0,1])")
    print("="*60)

    print("\n── Topics & Top Words ──")
    for i, tid in enumerate(valid_topic_ids):
        ws        = topic_model.get_topic(tid)
        top_words = [w for w, _ in ws[:10]] if ws else []
        coh_val   = per_topic_coh[i] if i < len(per_topic_coh) else "n/a"
        size_row  = topic_info.loc[topic_info["Topic"] == tid, "Count"]
        size      = int(size_row.values[0]) if len(size_row) > 0 else "?"
        coh_str   = f"{coh_val:.4f}" if isinstance(coh_val, float) else coh_val
        print(f"\n  Topic {tid:2d}  (n={size}, coherence={coh_str})")
        print(f"    Words: {', '.join(top_words)}")


    # ── 10. Save JSON ─────────────────────────────────────────────────────────
    results = {
        "config": {
            "embedding_model": EMBEDDING_MODEL,
            "umap_n_components": UMAP_N_COMPONENTS,
            "umap_n_neighbors": UMAP_N_NEIGHBORS,
            "optimal_k": OPTIMAL_K,
            "best_k_elbow": best_k_elbow,
            "best_k_silhouette": best_k_sil,
        },
        "evaluation": {
            "silhouette_score": round(sil_final, 4),
            "betacv": round(betacv, 4),
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
            "topic_id": int(tid),
            "size": size,
            "coherence_cv": round(coh_val, 4) if isinstance(coh_val, float) else None,
            "top_words": tw,
        })

    with open(OUTPUT_TOPICS, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved → {OUTPUT_TOPICS}")
    print(f"Plots saved   → {OUTPUT_PLOTS}")
    print("\nDone ✓")
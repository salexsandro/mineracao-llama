"""
Pipeline de Topic Modeling com BERTopic
Para mapeamento de hábitos de uso de short-form videos em relatos do Reddit (PT-BR)

Dependências:
    pip install bertopic sentence-transformers umap-learn hdbscan
    pip install plotly pandas numpy scikit-learn

Uso:
    python topic_modeling.py --input dataset.jsonl --output resultados/

Modelo de embeddings padrão: rufimelo/bert-large-portuguese-cased-sts
  - Treinado para similaridade semântica em português
  - ~1.3GB VRAM (confortável para RTX 4050 6GB)
  - Superior a modelos multilíngues genéricos para PT-BR
"""

import json
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

# KMeans como alternativa (ver seção CONFIGURAÇÃO DE CLUSTERING abaixo)
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO
# ─────────────────────────────────────────────

def carregar_relatos(caminho: str) -> pd.DataFrame:
    """
    Carrega o JSONL e filtra apenas os posts classificados como relato.
    Concatena title + body para dar mais contexto ao modelo.
    """
    registros = []
    with open(caminho, "r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            obj = json.loads(linha)

            # Filtra apenas relatos confirmados pelo Phi3
            classificacao = obj.get("classificacao_triagem", {})
            if not classificacao.get("e_relato", False):
                continue

            title = obj.get("title", "") or ""
            body  = obj.get("body",  "") or ""

            # Concatena título e corpo — o título costuma resumir o hábito principal
            texto_completo = f"{title.strip()} {body.strip()}".strip()

            if len(texto_completo) < 30:   # descarta textos muito curtos
                continue

            registros.append({
                "post_id":    obj.get("post_id", ""),
                "subreddit":  obj.get("subreddit", ""),
                "date":       obj.get("date", ""),
                "score":      obj.get("score", 0),
                "texto":      texto_completo,
                "justificativa": classificacao.get("justificativa", ""),
            })

    df = pd.DataFrame(registros)
    log.info(f"Relatos carregados: {len(df)}")
    return df


# ─────────────────────────────────────────────
# 2. GERAÇÃO DE EMBEDDINGS
# ─────────────────────────────────────────────

def gerar_embeddings(textos: list[str], model_name: str) -> np.ndarray:
    """
    Gera embeddings semânticos com sentence-transformers.

    Modelos recomendados para PT-BR (em ordem de preferência):
      - "rufimelo/bert-large-portuguese-cased-sts"     → melhor para PT-BR, ~1.3GB VRAM  ← PADRÃO
      - "neuralmind/bert-base-portuguese-cased"        → menor, mais rápido, ~450MB VRAM
      - "paraphrase-multilingual-mpnet-base-v2"        → multilíngue robusto, ~1.1GB VRAM
      - "paraphrase-multilingual-MiniLM-L12-v2"        → multilíngue leve, ~470MB VRAM

    NÃO use modelos generativos (Phi3, LLaMA, etc.) aqui — eles são decoders e
    não produzem embeddings semânticos adequados para clustering.
    """
    log.info(f"Carregando modelo de embeddings: {model_name}")
    modelo = SentenceTransformer(model_name)

    log.info("Gerando embeddings (isso pode levar alguns minutos)...")
    embeddings = modelo.encode(
        textos,
        show_progress_bar=True,
        batch_size=32,          # 32 é seguro para bert-large em 6GB VRAM; aumente para 64 com modelos base
        device="cuda",          # troca para "cpu" se não tiver GPU
        normalize_embeddings=True,  # normalização L2 melhora similaridade cosine
    )
    log.info(f"Embeddings gerados: shape {embeddings.shape}")
    return embeddings


# ─────────────────────────────────────────────
# 3. CONFIGURAÇÃO DOS COMPONENTES DO BERTOPIC
# ─────────────────────────────────────────────

def construir_modelo_bertopic(n_topics_kmeans: int = None) -> BERTopic:
    """
    Monta o BERTopic com os subcomponentes configurados.

    Parâmetros
    ----------
    n_topics_kmeans : int ou None
        Se None  → usa HDBSCAN (recomendado: detecta nº de tópicos automaticamente)
        Se int   → usa KMeans com esse número fixo de clusters
    """

    # ── 3a. Redução de dimensionalidade ──────────────────────────────────
    # UMAP projeta os embeddings de alta dimensão para um espaço menor
    # antes do clustering, preservando estrutura local e global.
    umap_model = UMAP(
        n_neighbors=15,     # vizinhos considerados — aumentar = tópicos mais amplos
        n_components=5,     # dimensões do espaço reduzido (5 é padrão BERTopic)
        min_dist=0.0,       # 0.0 força os pontos a ficarem mais agrupados
        metric="cosine",    # cosine é ideal para embeddings de texto
        random_state=42,
    )

    # ── 3b. Clustering ────────────────────────────────────────────────────
    if n_topics_kmeans is None:
        # HDBSCAN — padrão recomendado
        # min_cluster_size: mínimo de posts para formar um tópico
        # Ajuste: datasets menores → valor menor (ex: 5); maiores → 15~30
        cluster_model = HDBSCAN(
            min_cluster_size=10,
            min_samples=5,          # controla robustez contra ruído
            metric="euclidean",
            cluster_selection_method="eom",   # "leaf" para clusters menores
            prediction_data=True,             # necessário para transform()
        )
        log.info("Clustering: HDBSCAN (automático)")
    else:
        # KMeans — use quando quiser forçar um número específico de tópicos
        # Útil se você já tem categorias de hábitos em mente
        cluster_model = KMeans(
            n_clusters=n_topics_kmeans,
            random_state=42,
            n_init="auto",
        )
        log.info(f"Clustering: KMeans com {n_topics_kmeans} clusters")

    # ── 3c. Vetorizador de palavras-chave ─────────────────────────────────
    # CountVectorizer define quais n-gramas serão usados para representar tópicos
    # Stop words PT-BR: removemos conectivos e palavras funcionais que não
    # carregam semântica de hábito (ex: "que", "isso", "muito").
    # Palavras de domínio relevantes ("tiktok", "reels", "shorts") são mantidas.
    STOP_WORDS_PTBR = [
        "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "é",
        "com", "uma", "os", "no", "se", "na", "por", "mais", "as", "dos",
        "como", "mas", "foi", "ao", "ele", "das", "tem", "à", "seu", "sua",
        "ou", "ser", "quando", "muito", "há", "nos", "já", "está", "eu",
        "também", "só", "pelo", "pela", "até", "isso", "ela", "entre",
        "era", "depois", "sem", "mesmo", "aos", "ter", "seus", "quem",
        "nas", "me", "esse", "eles", "estão", "você", "tinha", "foram",
        "essa", "num", "nem", "suas", "meu", "às", "minha", "têm", "numa",
        "pelos", "pelas", "qual", "nós", "lhe", "deles", "essas", "esses",
        "pois", "aqui", "então", "bem", "agora", "porque", "nao", "não",
        "fazer", "ficou", "acho", "cada", "todo", "toda", "sempre", "nunca",
    ]

    vectorizer = CountVectorizer(
        ngram_range=(1, 2),         # unigrams e bigrams (ex: "rolar feed", "doom scroll")
        stop_words=STOP_WORDS_PTBR,
        min_df=3,                   # ignora termos que aparecem em < 3 documentos
        max_df=0.85,                # ignora termos em > 85% dos documentos
    )

    # ── 3d. c-TF-IDF com redução de outliers ─────────────────────────────
    ctfidf = ClassTfidfTransformer(
        reduce_frequent_words=True  # penaliza palavras muito comuns entre tópicos
    )

    # ── 3e. Representação com KeyBERT ─────────────────────────────────────
    # Refina os termos representativos usando similaridade semântica,
    # resultando em keywords mais coesas e descritivas por tópico
    representation_model = KeyBERTInspired()

    # ── 3f. Montagem final ────────────────────────────────────────────────
    modelo = BERTopic(
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf,
        representation_model=representation_model,
        top_n_words=15,             # palavras-chave por tópico
        min_topic_size=10,          # tópicos com menos posts são marcados como ruído
        nr_topics="auto",           # mescla tópicos muito similares automaticamente
        calculate_probabilities=True,
        verbose=True,
    )

    return modelo


# ─────────────────────────────────────────────
# 4. TREINAMENTO E ANÁLISE
# ─────────────────────────────────────────────

def treinar_e_analisar(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    modelo: BERTopic,
    output_dir: Path,
):
    textos = df["texto"].tolist()

    log.info("Treinando BERTopic...")
    topicos, probabilidades = modelo.fit_transform(textos, embeddings)

    df["topico_id"]  = topicos
    df["topico_prob"] = probabilidades.max(axis=1) if probabilidades.ndim > 1 else probabilidades

    # ── 4a. Resumo dos tópicos ────────────────────────────────────────────
    info_topicos = modelo.get_topic_info()
    log.info(f"\nTópicos encontrados: {len(info_topicos) - 1} (excluindo ruído -1)")
    print("\n" + info_topicos.to_string(index=False))

    # ── 4b. Detalhes por tópico (palavras-chave + posts representativos) ──
    relatorio = []
    for _, linha in info_topicos.iterrows():
        tid = linha["Topic"]
        if tid == -1:
            continue  # -1 = outliers no HDBSCAN

        keywords = [palavra for palavra, _ in modelo.get_topic(tid)]

        # Posts deste tópico, ordenados por probabilidade
        posts_topico = (
            df[df["topico_id"] == tid]
            .sort_values("topico_prob", ascending=False)
            .head(5)  # top 5 mais representativos
        )

        relatorio.append({
            "topico_id":        tid,
            "n_documentos":     int(linha["Count"]),
            "keywords":         keywords,
            "posts_exemplares": posts_topico[["post_id", "texto", "topico_prob"]].to_dict("records"),
        })

    # ── 4c. Persistência ──────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset com tópicos atribuídos
    df.to_json(output_dir / "posts_com_topicos.jsonl", orient="records", lines=True, force_ascii=False)
    log.info(f"Posts com tópicos salvos em: {output_dir / 'posts_com_topicos.jsonl'}")

    # Relatório de tópicos
    with open(output_dir / "relatorio_topicos.json", "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)
    log.info(f"Relatório salvo em: {output_dir / 'relatorio_topicos.json'}")

    # Modelo serializado (para reusar na etapa LLM)
    modelo.save(str(output_dir / "bertopic_model"), serialization="pytorch", save_ctfidf=True)
    log.info(f"Modelo salvo em: {output_dir / 'bertopic_model'}")

    # ── 4d. Visualizações HTML interativas ───────────────────────────────
    try:
        fig_topicos = modelo.visualize_topics()
        fig_topicos.write_html(str(output_dir / "viz_topicos.html"))

        fig_hierarquia = modelo.visualize_hierarchy()
        fig_hierarquia.write_html(str(output_dir / "viz_hierarquia.html"))

        fig_heatmap = modelo.visualize_heatmap()
        fig_heatmap.write_html(str(output_dir / "viz_heatmap.html"))

        fig_barchart = modelo.visualize_barchart(top_n_topics=20)
        fig_barchart.write_html(str(output_dir / "viz_keywords_por_topico.html"))

        log.info("Visualizações salvas em HTML.")
    except Exception as e:
        log.warning(f"Visualizações não geradas: {e}")

    return df, relatorio, info_topicos


# ─────────────────────────────────────────────
# 5. ESTATÍSTICAS DE SAÍDA
# ─────────────────────────────────────────────

def imprimir_estatisticas(df: pd.DataFrame, info_topicos: pd.DataFrame):
    total        = len(df)
    outliers     = (df["topico_id"] == -1).sum()
    classificados = total - outliers

    print("\n" + "═" * 55)
    print("  RESUMO DO PIPELINE")
    print("═" * 55)
    print(f"  Total de relatos:          {total}")
    print(f"  Classificados em tópicos:  {classificados} ({100*classificados/total:.1f}%)")
    print(f"  Outliers (ruído):          {outliers} ({100*outliers/total:.1f}%)")
    print(f"  Tópicos encontrados:       {len(info_topicos) - 1}")
    print("═" * 55)

    print("\nDistribuição por tópico:")
    dist = (
        df[df["topico_id"] != -1]
        .groupby("topico_id")
        .size()
        .sort_values(ascending=False)
    )
    for tid, count in dist.items():
        print(f"  Tópico {tid:3d}: {count:4d} posts  ({100*count/total:.1f}%)")


# ─────────────────────────────────────────────
# 6. ENTRYPOINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BERTopic para hábitos de uso de short-form videos")
    parser.add_argument("--input",   required=True,  help="Caminho para o arquivo .jsonl")
    parser.add_argument("--output",  default="resultados", help="Diretório de saída")
    parser.add_argument(
        "--embedding-model",
        default="rufimelo/bert-large-portuguese-cased-sts",
        help="Modelo sentence-transformers para embeddings (padrão: otimizado para PT-BR)"
    )
    parser.add_argument(
        "--kmeans",
        type=int,
        default=None,
        help="Se informado, usa KMeans com esse número de clusters em vez de HDBSCAN"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)

    # 1. Carregar
    df = carregar_relatos(args.input)
    if df.empty:
        log.error("Nenhum relato encontrado. Verifique o campo 'e_relato' no JSONL.")
        return

    # 2. Embeddings
    embeddings = gerar_embeddings(df["texto"].tolist(), args.embedding_model)

    # 3. Modelo
    modelo = construir_modelo_bertopic(n_topics_kmeans=args.kmeans)

    # 4. Treinar e salvar
    df, relatorio, info_topicos = treinar_e_analisar(df, embeddings, modelo, output_dir)

    # 5. Stats
    imprimir_estatisticas(df, info_topicos)

    print(f"\n✓ Pipeline concluído. Resultados em: {output_dir.resolve()}")
    print(  "  Próximo passo: use relatorio_topicos.json como entrada para o extrator LLM.")


if __name__ == "__main__":
    main()
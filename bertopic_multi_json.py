import pandas as pd
import glob
import os
import re
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from hdbscan import HDBSCAN

# ==========================================
# CONFIGURAÇÃO DE CAMINHOS ABSOLUTOS
# ==========================================
DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
PASTA_DADOS = os.path.join(DIRETORIO_ATUAL, "dados_classificados")
OUTPUT_CSV = os.path.join(DIRETORIO_ATUAL, "dataset_unsupervised_relatos_consolidado.csv")

# ==========================================
# 1. LISTA COMPLETA E RIGOROSA DE STOPWORDS DE DOMÍNIO
# ==========================================
DOMAIN_STOPWORDS = [
    "tiktok", "reels", "shorts", "youtube", "video", "videos", 
    "app", "apps", "phone", "scroll", "scrolling", "doomscrolling", 
    "reddit", "instagram", "social", "media", "sub", "platform", "cellphone",
    # Contrações comuns do Reddit separadas por pontuação (ex: don't -> don)
    "don", "isn", "wasn", "weren", "didn", "doesn", "haven", "hasn", "hadn", "won", "wouldn", "can", "couldn", "shouldn",
    # Muletas linguísticas e verbos mecânicos de altíssima frequência em relatos
    "just", "like", "time", "said", "told", "know", "feel", "really", "want", "think", "people", "ve", "im", "post", "posts", "watch", "watching", "content", "comment", "comments",
    "life"
]

# ==========================================
# 2. FUNÇÃO DE LIMPEZA TEXTUAL
# ==========================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'\[deleted\]|\[removed\]|\[excluído\]', '', text)
    text = re.sub(r'\n+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

# ==========================================
# 3. VARREDURA, LEITURA E FILTRAGEM EM LOTE
# ==========================================
print(f"Buscando arquivos na pasta interna: '{PASTA_DADOS}'...")

todos_arquivos = glob.glob(os.path.join(PASTA_DADOS, "*.jsonl"))
arquivos_jsonl = [f for f in todos_arquivos if os.path.basename(f).startswith("classificado_")]

if not arquivos_jsonl:
    raise FileNotFoundError(f"Nenhum arquivo começando com 'classificado_' encontrado em: {PASTA_DADOS}")

print(f"Sucesso! Encontrados {len(arquivos_jsonl)} arquivos classificados legítimos. Iniciando extração...")

lista_dfs = []

for caminho_arquivo in arquivos_jsonl:
    nome_arquivo = os.path.basename(caminho_arquivo)
    try:
        df_temp = pd.read_json(caminho_arquivo, lines=True)
        
        if 'classificacao_triagem' in df_temp.columns:
            # Extração segura da chave 'e_relato' de dentro do dicionário nativo
            df_temp['e_relato_bool'] = df_temp['classificacao_triagem'].apply(
                lambda x: bool(x.get('e_relato', False)) if isinstance(x, dict) else False
            )
            
            # Filtra mantendo apenas relatos verdadeiros
            df_filtrado = df_temp[df_temp['e_relato_bool'] == True].copy()
            print(f" -> {nome_arquivo}: Extraídos {len(df_filtrado)} relatos de {len(df_temp)} posts.")
            
            if not df_filtrado.empty:
                lista_dfs.append(df_filtrado)
        else:
            print(f" [AVISO] Coluna 'classificacao_triagem' ausente em {nome_arquivo}. Ignorando.")
            
    except Exception as e:
        print(f" [ERRO] Falha ao processar o arquivo {nome_arquivo}: {e}")

if not lista_dfs:
    raise ValueError("Nenhum relato (e_relato: true) pôde ser extraído após ler os arquivos.")

df = pd.concat(lista_dfs, ignore_index=True)
print(f"\nConsolidação concluída! Total de relatos reais carregados: {len(df)}")

# ==========================================
# 4. PREPARAÇÃO DO TEXTO DO DATASET
# ==========================================
if 'title' not in df.columns: df['title'] = ""
if 'body' not in df.columns: df['body'] = ""  

df['full_text'] = df['title'].fillna('') + " " + df['body'].fillna('')
df['full_text'] = df['full_text'].apply(clean_text)

df = df[df['full_text'].str.len() > 120].reset_index(drop=True)
print(f"Registros qualificados para o BERT (tamanho > 120 caracteres): {len(df)}")

# ==========================================
# 5. CONFIGURAÇÃO DOS COMPONENTES E STOPWORDS DO VALIDADOR
# ==========================================
docs = df['full_text'].tolist()

# Modelo de embedding semântico
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Clusterização via HDBSCAN
min_cluster = 150 if len(df) > 50000 else 50
hdbscan_model = HDBSCAN(
    min_cluster_size=min_cluster,
    min_samples=25,
    prediction_data=True
)

# CRÍTICO: Construção da união de Stopwords nativas e de domínio injetadas no construtor
# Suporta com segurança diferentes versões do scikit-learn
if hasattr(text, 'SET_WORDS'):
    stop_words_completas = list(text.SET_WORDS.union(DOMAIN_STOPWORDS))
else:
    stop_words_completas = list(set(text.ENGLISH_STOP_WORDS).union(DOMAIN_STOPWORDS))

# Passagem direta e explícita no construtor para forçar a filtragem no c-TF-IDF
vectorizer_model = CountVectorizer(stop_words=stop_words_completas)

# ==========================================
# 6. EXECUÇÃO DO MODELO NÃO SUPERVISIONADO
# ==========================================
print("\nIniciando Modelagem de Tópicos (BERTopic)...")
topic_model = BERTopic(
    embedding_model=embedding_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model, # Vincula o vetorizador blindado
    language="english",
    calculate_probabilities=False,
    nr_topics="auto" 
)

topics, probs = topic_model.fit_transform(docs)

# ==========================================
# 7. ENRIQUECIMENTO DO DATASET E SALVAMENTO
# ==========================================
df['topic_id'] = topics

topic_info = topic_model.get_topic_info()
df = df.merge(topic_info[['Topic', 'Name']], left_on='topic_id', right_on='Topic', how='left')
df = df.drop(columns=['Topic', 'e_relato_bool']) 
df.rename(columns={'Name': 'topic_name'}, inplace=True)

print("\n--- TOP 15 TÓPICOS GLOBAIS ENCONTRADOS (REFINADOS) ---")
print(topic_model.get_topic_info()[['Topic', 'Count', 'Name']].head(15))

df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
print(f"\nSucesso total! O dataset consolidado e limpo foi salvo em: {OUTPUT_CSV}")
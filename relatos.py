import json
import os
from openai import OpenAI

# ==========================================
# CONFIGURAÇÕES
# ==========================================
INPUT_FILE = "reddit_clean_data_35k.jsonl"
OUTPUT_FILE = "classificacoes.jsonl"
BATCH_SIZE = 20
MODEL = "gpt-4o-mini"

# Inicializa o cliente da OpenAI 
# (Ele buscará automaticamente a variável de ambiente OPENAI_API_KEY)
client = OpenAI()

# Adaptei ligeiramente seu prompt para que ele processe uma lista de posts 
# e retorne um objeto JSON contendo um array (exigência do json_object da OpenAI)
SYSTEM_PROMPT = """
You are an expert annotator studying short-form video consumption.

Your task is to classify Reddit posts that mention:
* TikTok
* YouTube Shorts
* Instagram Reels
* short-form video feeds
* scrolling behavior related to these platforms

Assign exactly one category per post.

CATEGORY 0: NOT_REPORT
The post does not contain a personal experience.
Examples: News, Research discussion, General opinions, Memes, Advice, Questions without self-disclosure.

CATEGORY 1: UNRELATED_REPORT
The post contains a personal experience, but short-video consumption is not the subject of the report.
The platform is merely: a source of information, a discovery channel, a recommendation source, or a communication platform.

CATEGORY 2: RELEVANT_REPORT
The post contains a personal report involving short-video consumption.
This includes: watching videos, scrolling behavior, excessive use, quitting, reducing use, relapsing, cravings, habits, benefits, harms, effects, outcomes attributed to use.

Decision rules:
1. The author must describe their own experience for CATEGORY 1 or CATEGORY 2.
2. If TikTok, Shorts, or Reels are only where the author learned something, use CATEGORY 1.
3. If the post discusses the author's consumption, use CATEGORY 2.
4. If the post discusses effects, benefits, harms, urges, quitting, habits, or usage patterns related to consumption, use CATEGORY 2.

Return JSON only. Do not add markdown formatting or conversational notes.
You will receive a list of posts. You must return a JSON object with a single key "results", containing a list of dictionaries.

Schema for each object in the "results" list:
{
  "post_id": "<string: copy exactly from input>",
  "category": <int: choose 0, 1, or 2>,
  "label": "<string: choose NOT_REPORT, UNRELATED_REPORT, or RELEVANT_REPORT>",
  "confidence": <float: between 0.00 and 1.00>,
  "reason": "<string: brief explanation for the decision>"
}
"""

def carregar_ids_processados(caminho_arquivo):
    """Lê o arquivo de saída para identificar os post_ids que já foram classificados."""
    ids_processados = set()
    if os.path.exists(caminho_arquivo):
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            for linha in f:
                linha = linha.strip()
                if linha:
                    try:
                        dado = json.loads(linha)
                        if "post_id" in dado:
                            ids_processados.add(dado["post_id"])
                    except json.JSONDecodeError:
                        continue
    return ids_processados

def carregar_posts_pendentes(caminho_arquivo, ids_processados):
    """Lê o input e retorna apenas os posts que ainda não foram processados."""
    posts_pendentes = []
    if not os.path.exists(caminho_arquivo):
        print(f"Arquivo {caminho_arquivo} não encontrado.")
        return posts_pendentes

    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        for linha in f:
            linha = linha.strip()
            if linha:
                try:
                    post = json.loads(linha)
                    if post.get("post_id") not in ids_processados:
                        posts_pendentes.append(post)
                except json.JSONDecodeError:
                    continue
    return posts_pendentes

def criar_lotes(lista, tamanho_lote):
    """Gerador que divide a lista em lotes do tamanho especificado."""
    for i in range(0, len(lista), tamanho_lote):
        yield lista[i:i + tamanho_lote]

def processar_dados():
    ids_processados = carregar_ids_processados(OUTPUT_FILE)
    posts_pendentes = carregar_posts_pendentes(INPUT_FILE, ids_processados)

    print(f"Resumo:")
    print(f"- Posts já classificados: {len(ids_processados)}")
    print(f"- Posts aguardando processamento: {len(posts_pendentes)}")
    print("-" * 30)

    if not posts_pendentes:
        print("Nenhum novo post para classificar. Encerrando.")
        return

    # Abre o arquivo de saída no modo "append" (adicionar ao final)
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as out_file:
        for lote in criar_lotes(posts_pendentes, BATCH_SIZE):
            
            # Filtramos os campos que enviamos à LLM para economizar tokens
            # Não há necessidade de enviar URL, data, número de comentários, etc.
            lote_para_llm = [
                {
                    "post_id": p["post_id"],
                    "title": p.get("title", ""),
                    "body": p.get("body", "")
                }
                for p in lote
            ]
            
            print(f"Enviando lote com {len(lote_para_llm)} posts para {MODEL}...")
            print(f"- Posts já classificados: {len(ids_processados)}")
            print(f"- Posts aguardando processamento: {len(posts_pendentes)}")
            
            try:
                resposta = client.chat.completions.create(
                    model=MODEL,
                    response_format={"type": "json_object"},
                    temperature=0.0, # Temperatura 0 para garantir consistência na anotação
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps({"posts": lote_para_llm})}
                    ]
                )
                
                conteudo_resposta = resposta.choices[0].message.content
                dados_json = json.loads(conteudo_resposta)
                
                # O json retornado deve ter a chave "results" devido ao prompt ajustado
                resultados = dados_json.get("results", [])
                
                for res in resultados:
                    # Escreve imediatamente no disco. Se o script cair, o que foi salvo não se perde.
                    out_file.write(json.dumps(res) + '\n')
                    # Atualiza o buffer do arquivo
                    out_file.flush()
                
                print("Lote salvo com sucesso.")
                
            except Exception as e:
                print(f"Erro ao processar o lote: {e}")
                print("Interrompendo a execução. Corrija o erro e rode o script novamente para continuar de onde parou.")
                break

if __name__ == "__main__":
    processar_dados()
import json
import os
import traceback
from openai import OpenAI, APIError, APIConnectionError, RateLimitError, AuthenticationError

# ==========================================
# CONFIGURAÇÕES
# ==========================================
INPUT_FILE = "reddit_clean_data_35k.jsonl"
OUTPUT_FILE = "classificacoes.jsonl"
BATCH_SIZE = 20
MODEL = "gpt-4o-mini"

# Inicializa o cliente da OpenAI
client = OpenAI()

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
    for i in range(0, len(lista), tamanho_lote):
        yield lista[i:i + tamanho_lote]

def exibir_barra_progresso(atual, total, tamanho_barra=40):
    """Exibe uma barra de progresso visual no terminal."""
    progresso = atual / total
    preenchido = int(tamanho_barra * progresso)
    barra = '█' * preenchido + '-' * (tamanho_barra - preenchido)
    porcentagem = progresso * 100
    print(f"Progresso da Sessão: |{barra}| {atual}/{total} ({porcentagem:.1f}%)")
    print("-" * 50)

def processar_dados():
    ids_processados = carregar_ids_processados(OUTPUT_FILE)
    posts_pendentes = carregar_posts_pendentes(INPUT_FILE, ids_processados)

    total_pendentes = len(posts_pendentes)
    processados_agora = 0

    print(f"\nResumo Inicial:")
    print(f"- Posts já classificados anteriormente: {len(ids_processados)}")
    print(f"- Posts aguardando processamento agora: {total_pendentes}")
    print("=" * 50)

    if not posts_pendentes:
        print("Nenhum novo post para classificar. Encerrando.")
        return

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as out_file:
        for lote in criar_lotes(posts_pendentes, BATCH_SIZE):
            
            lote_para_llm = [
                {
                    "post_id": p["post_id"],
                    "title": p.get("title", ""),
                    "body": p.get("body", "")
                }
                for p in lote
            ]
            
            print(f"Enviando lote de {len(lote_para_llm)} posts para a API...")
            
            try:
                resposta = client.chat.completions.create(
                    model=MODEL,
                    response_format={"type": "json_object"},
                    temperature=0.0, 
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps({"posts": lote_para_llm})}
                    ]
                )
                
                conteudo_resposta = resposta.choices[0].message.content
                dados_json = json.loads(conteudo_resposta)
                resultados = dados_json.get("results", [])
                
                for res in resultados:
                    out_file.write(json.dumps(res) + '\n')
                out_file.flush()
                
                # Atualiza os contadores e exibe a barra de progresso
                processados_agora += len(lote)
                exibir_barra_progresso(processados_agora, total_pendentes)
                
            except APIConnectionError as e:
                print("\n[ERRO DE CONEXÃO] Falha na comunicação com a rede ou servidores da OpenAI.")
                print(f"Detalhes: {e}")
                break
            except RateLimitError as e:
                print("\n[ERRO DE RATE LIMIT] Limite de requisições excedido ou cota da API esgotada.")
                print(f"Detalhes: {e}")
                break
            except AuthenticationError as e:
                print("\n[ERRO DE AUTENTICAÇÃO] A API Key fornecida é inválida ou não tem permissões suficientes.")
                print(f"Detalhes: {e}")
                break
            except APIError as e:
                print(f"\n[ERRO NA API] A OpenAI retornou um erro genérico do lado do servidor (Status HTTP: {e.status_code}).")
                print(f"Detalhes: {e.message}")
                break
            except json.JSONDecodeError as e:
                print("\n[ERRO DE PARSING JSON] A LLM não retornou um JSON estruturado de forma válida.")
                print(f"Detalhes: {e}")
                print(f"Conteúdo bruto recebido:\n{conteudo_resposta}")
                break
            except Exception as e:
                print("\n[ERRO DESCONHECIDO] Ocorreu uma exceção não catalogada no sistema.")
                print("Stacktrace completo:")
                traceback.print_exc()
                break

if __name__ == "__main__":
    processar_dados()
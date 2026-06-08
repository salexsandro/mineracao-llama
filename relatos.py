import os
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
import re

# ==========================================
# CONFIGURAÇÃO DE AMBIENTE E CAMINHOS
# ==========================================
DIRETORIO_INPUT = "./"
DIRETORIO_OUTPUT = "./"
NOME_MODELO = "gpt-4o-mini"

TAMANHO_DO_LOTE = 20  # Quantas linhas processar antes de salvar no disco
NUM_WORKERS = 5       # Quantas requisições simultâneas fazer dentro do lote

MODO_TESTE = False
LIMITE_POSTS_TESTE = 100

api_key_env = os.environ.get("api_key")

if not api_key_env:
    print("❌ Erro: Variável de ambiente 'api_key' não encontrada.")
    sys.exit(1)

client = OpenAI(api_key=api_key_env)

# ==========================================
# PROMPT DE CLASSIFICAÇÃO
# ==========================================
PROMPT_CLASSIFICACAO = """
You are an expert annotator studying short-form video consumption.
Your task is to classify Reddit posts that mention:
* TikTok
* YouTube Shorts
* Instagram Reels
* short-form video feeds
* scrolling behavior related to these platforms

Assign exactly one category.

CATEGORY 0: NOT_REPORT
The post does not contain a personal experience.
Examples:
* News
* Research discussion
* General opinions
* Memes
* Advice
* Questions without self-disclosure

CATEGORY 1: UNRELATED_REPORT
The post contains a personal experience, but short-video consumption is not the subject of the report.
The platform is merely:
* a source of information
* a discovery channel
* a recommendation source
* a communication platform
Examples:
* "TikTok helped me discover I have ADHD."
* "I learned a study technique from TikTok."
* "I found a recipe on TikTok."

CATEGORY 2: RELEVANT_REPORT
The post contains a personal report involving short-video consumption.
This includes:
* watching videos
* scrolling behavior
* excessive use
* quitting
* reducing use
* relapsing
* cravings
* habits
* benefits
* harms
* effects
* outcomes attributed to use
Examples:
* "I spend 5 hours a day on TikTok."
* "Watching Shorts hurts my concentration."
* "Deleting Reels improved my sleep."
* "I keep reinstalling TikTok."
* "TikTok helps me relax."

Decision rules:
1. The author must describe their own experience for CATEGORY 1 or CATEGORY 2.
2. If TikTok, Shorts, or Reels are only where the author learned something, use CATEGORY 1.
3. If the post discusses the author's consumption, use CATEGORY 2.
4. If the post discusses effects, benefits, harms, urges, quitting, habits, or usage patterns related to consumption, use CATEGORY 2.

Return JSON only. Do not add markdown formatting or conversational notes.
Schema:
{
  "post_id": "",
  "category": 0,
  "label": "NOT_REPORT",
  "confidence": 0.00
}
"""

def extrair_json_via_regex(texto_resposta):
    if not texto_resposta:
        return None
    try:
        match = re.search(r'\{.*\}', texto_resposta, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass
    return None

def classificar_post(post_id, title, body):
    max_tentativas = 4
    for tentativa in range(max_tentativas):
        try:
            user_content = f"POST_ID:\n{post_id}\n\nTITLE:\n{title}\n\nBODY:\n{body}"

            res = client.chat.completions.create(
                model=NOME_MODELO,
                messages=[
                    {"role": "system", "content": PROMPT_CLASSIFICACAO},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            texto_gerado = res.choices[0].message.content
            dados_json = extrair_json_via_regex(texto_gerado)

            if dados_json:
                # Garante que o post_id está correto no resultado
                dados_json["post_id"] = post_id
                return dados_json

        except Exception as e:
            msg_erro = str(e).lower()
            if "rate limit" in msg_erro or "429" in msg_erro or "timeout" in msg_erro:
                print(f"\n[Motivo do Bloqueio da OpenAI] {e}")
                tempo_espera = 2 ** tentativa
                print(f"\n[Rate Limit] Esperando {tempo_espera}s no post {post_id}...")
                time.sleep(tempo_espera)
            else:
                print(f"\n❌ ERRO FATAL no ID {post_id}: {type(e).__name__} - {e}")
                break

    # Fallback caso todas as tentativas falhem
    print(f"\n[Aviso] Retornando fallback para o ID {post_id}")
    return {
        "post_id": post_id,
        "category": -1,
        "label": "ERROR",
        "confidence": 0.0,
        "reason": "Falha após múltiplas tentativas"
    }

def carregar_ids_processados(caminho_saida):
    ids_processados = set()
    if os.path.exists(caminho_saida):
        with open(caminho_saida, 'r', encoding='utf-8') as f:
            for linha in f:
                if linha.strip():
                    try:
                        dado = json.loads(linha)
                        if "post_id" in dado:
                            ids_processados.add(str(dado["post_id"]))
                    except:
                        pass
    return ids_processados

def processar_arquivo_jsonl(caminho_entrada, caminho_saida, nome_arquivo):
    print(f"\nIniciando leitura prévia de: {nome_arquivo}")

    ids_ja_processados = carregar_ids_processados(caminho_saida)
    if ids_ja_processados:
        print(f"🔄 Checkpoint encontrado: {len(ids_ja_processados)} posts já classificados serão pulados.")

    linhas_para_processar = []

    with open(caminho_entrada, 'r', encoding='utf-8') as infile:
        for index, linha in enumerate(infile):
            if linha.strip():
                try:
                    dado = json.loads(linha)
                    post_id = str(dado.get("post_id", str(index)))

                    if post_id not in ids_ja_processados:
                        title = dado.get("title", "")
                        body = dado.get("body", "")

                        # Ignora posts com conteúdo muito curto
                        texto_completo = f"{title} {body}".strip()
                        if len(texto_completo) >= 10:
                            linhas_para_processar.append((post_id, title, body))
                except:
                    pass

    total_linhas = len(linhas_para_processar)
    if total_linhas == 0:
        print(f"✅ [{nome_arquivo}] Nenhum novo post para classificar.")
        return False

    if MODO_TESTE:
        linhas_para_processar = linhas_para_processar[:LIMITE_POSTS_TESTE]
        total_linhas = len(linhas_para_processar)
        print(f"⚠️ MODO DE TESTE ATIVO: Processando amostra de {total_linhas} posts.")

    print(f"🚀 Iniciando classificação em lotes de {TAMANHO_DO_LOTE} para {total_linhas} posts...\n")

    with open(caminho_saida, 'a', encoding='utf-8') as outfile:
        with tqdm(total=total_linhas, desc="Progresso Geral", unit="posts", ncols=100) as pbar:

            for i in range(0, total_linhas, TAMANHO_DO_LOTE):
                lote_atual = linhas_para_processar[i : i + TAMANHO_DO_LOTE]
                resultados_do_lote = []

                with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                    futures = {
                        executor.submit(classificar_post, item[0], item[1], item[2]): item
                        for item in lote_atual
                    }

                    for future in as_completed(futures):
                        resultado = future.result()
                        if resultado:
                            resultados_do_lote.append(resultado)
                        pbar.update(1)

                for res in resultados_do_lote:
                    outfile.write(json.dumps(res, ensure_ascii=False) + "\n")
                outfile.flush()

                time.sleep(1.0)

    print(f"\n🎉 Concluído: {nome_arquivo} classificado com sucesso!")
    return True

def pipeline_principal():
    if len(sys.argv) < 2:
        print("\n❌ Erro: Você esqueceu de passar o nome do arquivo!")
        print("Uso correto: python classificar_posts.py nome_do_arquivo.jsonl\n")
        sys.exit(1)

    arquivo_alvo = sys.argv[1]
    os.makedirs(DIRETORIO_OUTPUT, exist_ok=True)

    caminho_in = os.path.join(DIRETORIO_INPUT, arquivo_alvo)
    caminho_out = os.path.join(DIRETORIO_OUTPUT, f"classificado_{arquivo_alvo}")

    if not os.path.exists(caminho_in):
        print(f"\n❌ Erro: O arquivo '{arquivo_alvo}' não foi encontrado na pasta '{DIRETORIO_INPUT}'.")
        sys.exit(1)

    processar_arquivo_jsonl(caminho_in, caminho_out, arquivo_alvo)

if __name__ == "__main__":
    pipeline_principal()
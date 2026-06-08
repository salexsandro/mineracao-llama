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
LIMITE_POSTS_TESTE = 40  # Aumentado para testar 2 lotes de 20

api_key_env = os.environ.get("api_key")

if not api_key_env:
    print("❌ Erro: Variável de ambiente 'api_key' não encontrada.")
    sys.exit(1)

client = OpenAI(api_key=api_key_env)

# ==========================================
# PROMPT UNIFICADO
# ==========================================
PROMPT_UNIFICADO = """
You are a linguistic annotation tool. Extract clinical symptoms and usage habits from the user's text based on specific allowed tokens.

ALLOWED SYMPTOMS: "Attention deficit", "Anxiety", "Depression", "Anhedonia", "Insomnia", "Depersonalisation", "Irritability", "Guilt", "Cephalea", "Cognitive disorder", "Memory impairment", "Asthenopia", "Vision blurred", "Neck pain", "Pain in hand".

ALLOWED HABITS: "Salience", "Tolerance", "Mood Modification", "Relapse", "Conflict".

Return ONLY a raw JSON object matching this exact structure:
{
  "id": "<insert the provided ID here>",
  "sintomas": ["Token 1", "Token 2"],
  "habitos": ["Token 1", "Token 2"]
}
If no matches are found, use empty arrays []. Do not add markdown formatting or conversational notes.
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

def extrair_dados_otimizados(post_id, text_body):
    clean_body = re.sub(r'["\'\\]', '', text_body).replace('\n', ' ').strip()
    
    max_tentativas = 4
    for tentativa in range(max_tentativas):
        try:
            # 2. VERIFICAÇÃO ANTES DA API
            # print(f"Enviando post {post_id} para a API...") # Descomente se quiser ver o fluxo
            
            res = client.chat.completions.create(
                model=NOME_MODELO,
                messages=[
                    {"role": "system", "content": PROMPT_UNIFICADO},
                    {"role": "user", "content": f"ID: {post_id}\nTEXT:\n\"{clean_body}\""}
                ],
                temperature=0.0,
                response_format={"type": "json_object"} 
            )
            
            texto_gerado = res.choices[0].message.content
            dados_json = extrair_json_via_regex(texto_gerado)
            
            if dados_json:
                return dados_json
                
        except Exception as e:
            msg_erro = str(e).lower()
            if "rate limit" in msg_erro or "429" in msg_erro or "timeout" in msg_erro:
                # ADICIONAMOS ESTA LINHA PARA LER A MENSAGEM REAL DA OPENAI
                print(f"\n[Motivo do Bloqueio da OpenAI] {e}") 
                tempo_espera = 2 ** tentativa 
                print(f"\n[Rate Limit] Esperando {tempo_espera}s no post {post_id}...")
                time.sleep(tempo_espera)
            else:
                # SE CAIR AQUI, VAMOS DESCOBRIR O MOTIVO!
                print(f"\n❌ ERRO FATAL no ID {post_id}: {type(e).__name__} - {e}")
                break # Sai do loop de tentativas e retorna o vazio

    # Fallback caso todas as tentativas falhem
    print(f"\n[Aviso] Retornando fallback vazio para o ID {post_id}")
    return {"id": post_id, "sintomas": [], "habitos": []}
def carregar_ids_processados(caminho_saida):
    ids_processados = set()
    if os.path.exists(caminho_saida):
        with open(caminho_saida, 'r', encoding='utf-8') as f:
            for linha in f:
                if linha.strip():
                    try:
                        dado = json.loads(linha)
                        if "id" in dado:
                            ids_processados.add(str(dado["id"]))
                    except:
                        pass
    return ids_processados

def processar_arquivo_jsonl(caminho_entrada, caminho_saida, nome_arquivo):
    print(f"\nIniciando leitura prévia de: {nome_arquivo}")
    
    ids_ja_processados = carregar_ids_processados(caminho_saida)
    if ids_ja_processados:
        print(f"🔄 Checkpoint encontrado: {len(ids_ja_processados)} posts já extraídos serão pulados.")

    linhas_para_processar = []

    with open(caminho_entrada, 'r', encoding='utf-8') as infile:
        for index, linha in enumerate(infile):
            if linha.strip():
                try:
                    dado = json.loads(linha)
                    
                    texto_titulo = dado.get("title", "")
                    texto_body = dado.get("body", "")
                    
                    # Concatena o título e o corpo com uma quebra de linha dupla
                    texto_completo = f"TITLE: {texto_titulo}\n\nBODY: {texto_body}".strip()
                    
                    if len(texto_completo) >= 30:
                        post_id = str(dado.get("post_id", str(index)))
                        
                        if post_id not in ids_ja_processados:
                            # Agora o item[1] será o texto completo (Título + Body)
                            linhas_para_processar.append((post_id, texto_completo))
                except: 
                    pass

    total_linhas = len(linhas_para_processar)
    if total_linhas == 0:
        print(f"✅ [{nome_arquivo}] Nenhum novo relato para processar.")
        return False

    if MODO_TESTE:
        linhas_para_processar = linhas_para_processar[:LIMITE_POSTS_TESTE]
        total_linhas = len(linhas_para_processar)
        print(f"⚠️ MODO DE TESTE ATIVO: Processando amostra de {total_linhas} relatos.")

    print(f"🚀 Iniciando extração em lotes de {TAMANHO_DO_LOTE} para {total_linhas} relatos...\n")

    # Abre o arquivo em modo Append (adicionar ao final)
    with open(caminho_saida, 'a', encoding='utf-8') as outfile:
        # Barra de progresso geral
        with tqdm(total=total_linhas, desc="Progresso Geral", unit="posts", ncols=100) as pbar:
            
            # Loop principal cortando a lista de 20 em 20
            for i in range(0, total_linhas, TAMANHO_DO_LOTE):
                lote_atual = linhas_para_processar[i : i + TAMANHO_DO_LOTE]
                resultados_do_lote = []

                # Dispara as 20 requisições usando o pool de threads
                with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                    futures = {
                        executor.submit(extrair_dados_otimizados, item[0], item[1]): item 
                        for item in lote_atual
                    }
                    
                    for future in as_completed(futures):
                        resultado = future.result()
                        if resultado:
                            resultados_do_lote.append(resultado)
                        pbar.update(1) # Atualiza a barra assim que cada post termina
                
                # Salva o lote inteiro no arquivo de uma vez só
                for res in resultados_do_lote:
                    outfile.write(json.dumps(res, ensure_ascii=False) + "\n")
                outfile.flush() # Força a gravação física no HD/SSD
                
                # Respiro entre os lotes para evitar block da API
                time.sleep(1.0) 

    print(f"\n🎉 Concluído: {nome_arquivo} atualizado com sucesso!")
    return True

def pipeline_principal():
    if len(sys.argv) < 2:
        print("\n❌ Erro: Você esqueceu de passar o nome do arquivo!")
        print("Uso correto: python nome_do_script.py nome_do_arquivo.jsonl\n")
        sys.exit(1)
        
    arquivo_alvo = sys.argv[1]
    os.makedirs(DIRETORIO_OUTPUT, exist_ok=True)
    
    caminho_in = os.path.join(DIRETORIO_INPUT, arquivo_alvo)
    caminho_out = os.path.join(DIRETORIO_OUTPUT, f"extraido_otimizado_{arquivo_alvo}")
    
    if not os.path.exists(caminho_in):
        print(f"\n❌ Erro: O arquivo '{arquivo_alvo}' não foi encontrado na pasta '{DIRETORIO_INPUT}'.")
        sys.exit(1)

    processar_arquivo_jsonl(caminho_in, caminho_out, arquivo_alvo)

if __name__ == "__main__":
    pipeline_principal()
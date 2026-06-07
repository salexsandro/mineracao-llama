import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from tqdm import tqdm 
import re 

# ==========================================
# CONFIGURAÇÃO DE AMBIENTE E CAMINHOS
# ==========================================
DIRETORIO_INPUT = "./dados_classificados" 
DIRETORIO_OUTPUT = "./dados_extracao_clinica"
NUM_CORES_TRABALHADORES = 1
NOME_MODELO = "phi3"

# ==========================================
# CONFIGURAÇÃO DO MODO DE TESTE CONTROLADO
# ==========================================
MODO_TESTE = False       
LIMITE_POSTS_TESTE = 10 

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# ==========================================
# PROMPTS RESTRUTURADOS PARA RESPOSTA LIVRE (TEXTO PURO)
# ==========================================
PROMPT_SINTOMAS = """
You are a factual linguistic annotation tool for social media mining. Your sole job is to translate user health complaints into standard medical terminology tokens.
Analyze the input text and extract any mentions where the user complains about their own mood, brain, eyes, head, or body issues. 

Map their colloquial symptoms STRICTLY to these allowed tokens:
- "Attention deficit" (can't focus, attention span ruined, brain fog, concentration gone, distracted)
- "Anxiety" (anxious, panic attacks, stressed, jittery, general dread, uneasy)
- "Depression" (depressed, sad, empty, hopeless, low mood, crying)
- "Anhedonia" (no joy, numb, dopamine fried, lost interest in hobbies, blank)
- "Insomnia" (can't sleep, stay up scrolling, tossing and turning at 3 AM, sleep deprivation)
- "Depersonalisation" (feeling like a zombie, brain rot, dissociation, autopilot, detached)
- "Irritability" (getting mad without the phone, easily annoyed, tech-withdrawal anger)
- "Guilt" (post-scrolling guilt, shame for wasting time, mad at myself)
- "Cephalea" (bad headache, migraine, head hurts)
- "Cognitive disorder" (severe brain fog, mental fatigue, fuzzy brain)
- "Memory impairment" (forgetting things easily, poor short-term memory)
- "Asthenopia" (eye strain, tired eyes, burning eyes, eyes hurt from screen)
- "Vision blurred" (blurry vision, sight goes fuzzy)
- "Neck pain" (neck hurts, tech neck, stiff neck)
- "Pain in hand" (finger cramps, thumb hurts from swiping, wrist pain)

You must respond by printing a raw JSON object with this exact key: {"meddra_preferred_terms": ["Token 1", "Token 2"]}. 
If no allowed symptoms are present, return {"meddra_preferred_terms": []}. Do not add any conversational notes.
"""

PROMPT_HABITOS = """
You are a factual linguistic annotation tool for social media mining. Your sole job is to translate user behavior descriptions into standard addiction tokens.
Analyze the input text and extract usage habits. 

Map them STRICTLY to these allowed tokens based on the Bergen Social Media Addiction Scale:
- "Salience" (opening app immediately upon waking up, thinking about the feed all day, craving the screen)
- "Tolerance" (spending hours scrolling, can't stop, watching more and more, losing track of time)
- "Mood Modification" (scrolling because bored, sad, stressed, lonely, or to avoid tasks/escape reality)
- "Relapse" (reinstalling the app after deleting it, failed to stay away, trying to quit but failing)
- "Conflict" (skipping sleep, skipping studies, homework, real-life duties, or hurting relationships to scroll)

You must respond by printing a raw JSON object with this exact key: {"bsmas_habits_detected": ["Token 1", "Token 2"]}. 
If no allowed habits are present, return {"bsmas_habits_detected": []}. Do not add any conversational notes.
"""

def extrair_json_via_regex(texto_resposta, chave_esperada):
    """Varre o texto livre gerado pelo modelo buscando um bloco JSON válido"""
    if not texto_resposta:
        return []
    try:
        match = re.search(r'\{.*\}', texto_resposta, re.DOTALL)
        if match:
            string_json = match.group()
            string_json = re.sub(r',\s*\]', ']', string_json)
            string_json = re.sub(r',\s*\}', '}', string_json)
            
            dados = json.loads(string_json)
            return dados.get(chave_esperada, [])
    except:
        pass
    return []

def extrair_entidades_clinicas(text_body):
    clean_body = re.sub(r'["\'\\]', '', text_body).replace('\n', ' ').strip()
    if len(clean_body) < 15:
        return {"bsmas_habits_detected": [], "meddra_preferred_terms": [], "causal_mapping": [], "text_evidence": ""}
    
    sintomas = []
    habitos = []

    # ---- ESTÁGIO 1: SINTOMAS ----
    try:
        res_sintomas = client.chat.completions.create(
            model=NOME_MODELO,
            messages=[
                {"role": "system", "content": PROMPT_SINTOMAS},
                {"role": "user", "content": f"INPUT TEXT:\n\"{clean_body}\""}
            ],
            temperature=0.0,
            extra_body={"options": {"num_ctx": 2048}} # Reseta o cache de contexto do Ollama
        )
        sintomas = extrair_json_via_regex(res_sintomas.choices[0].message.content, "meddra_preferred_terms")
    except Exception as e:
        tqdm.write(f" -> Falha Stage 1: {e}")

    # ---- ESTÁGIO 2: HÁBITOS ----
    try:
        res_habitos = client.chat.completions.create(
            model=NOME_MODELO,
            messages=[
                {"role": "system", "content": PROMPT_HABITOS},
                {"role": "user", "content": f"INPUT TEXT:\n\"{clean_body}\""}
            ],
            temperature=0.0,
            extra_body={"options": {"num_ctx": 2048}} # Reseta o cache de contexto do Ollama
        )
        habitos = extrair_json_via_regex(res_habitos.choices[0].message.content, "bsmas_habits_detected")
    except Exception as e:
        tqdm.write(f" -> Falha Stage 2: {e}")

    # ---- MAPEAMENTO CAUSAL EM PYTHON ----
    mapeamento = []
    if habitos and sintomas:
        for h in habitos:
            for s in sintomas:
                mapeamento.append({
                    "trigger_habit": h,
                    "consequence_symptom": s
                })

    return {
        "bsmas_habits_detected": habitos,
        "meddra_preferred_terms": sintomas,
        "causal_mapping": mapeamento,
        "text_evidence": clean_body[:150] + "..." if len(clean_body) > 150 else clean_body
    }

def processar_linha(linha, index, nome_arquivo):
    if not linha.strip(): 
        return None
    try:
        dado = json.loads(linha)
        triagem = dado.get("classificacao_triagem", {})
        e_relato = triagem.get("e_relato", False) if isinstance(triagem, dict) else False
        
        if not e_relato: return None 

        texto_body = dado.get("body", "")
        if not texto_body or len(str(texto_body)) < 30: return None

        dado["extracao_cientifica"] = extrair_entidades_clinicas(texto_body)
        return dado
    except:
        return None

def processar_arquivo_jsonl(caminho_entrada, caminho_saida, nome_arquivo):
    tqdm.write(f"\nIniciando processamento: {nome_arquivo}")
    linhas_para_processar = []

    with open(caminho_entrada, 'r', encoding='utf-8') as infile:
        for index, linha in enumerate(infile):
            if linha.strip():
                try:
                    dado = json.loads(linha)
                    triagem = dado.get("classificacao_triagem", {})
                    e_relato = triagem.get("e_relato", False) if isinstance(triagem, dict) else False
                    if e_relato:
                        linhas_para_processar.append((linha, index + 1))
                except: pass

    total_linhas = len(linhas_para_processar)
    if total_linhas == 0:
        tqdm.write(f"[{nome_arquivo}] Sem novos relatos qualificados.")
        return False

    if MODO_TESTE:
        linhas_para_processar = linhas_para_processar[:LIMITE_POSTS_TESTE]
        total_linhas = len(linhas_para_processar)
        tqdm.write(f"⚠️ MODO DE TESTE ATIVO: Processando amostra de {total_linhas} relatos.")

    resultados = [None] * total_linhas

    with ThreadPoolExecutor(max_workers=NUM_CORES_TRABALHADORES) as executor:
        futures = {
            executor.submit(processar_linha, item[0], item[1], nome_arquivo): i
            for i, item in enumerate(linhas_para_processar)
        }
        for future in tqdm(as_completed(futures), total=total_linhas, desc=nome_arquivo, unit="clinicos", ncols=100):
            posicao = futures[future]
            resultado_dado = future.result()
            if resultado_dado: resultados[posicao] = resultado_dado

    with open(caminho_saida, 'w', encoding='utf-8') as outfile:
        for res in resultados:
            if res: outfile.write(json.dumps(res, ensure_ascii=False) + "\n")

    tqdm.write(f"Concluído: {nome_arquivo} salvo e sobrescrito com sucesso.\n")
    return True

def as_completed(futures):
    from concurrent.futures import as_completed
    return as_completed(futures)

def pipeline_principal():
    # Verifica se o usuário passou o nome do arquivo como argumento no terminal
    if len(sys.argv) < 2:
        print("\n❌ Erro: Você esqueceu de passar o nome do arquivo!")
        print("Uso correto: python nome_do_script.py nome_do_arquivo.jsonl\n")
        sys.exit(1)
        
    arquivo_alvo = sys.argv[1]
    os.makedirs(DIRETORIO_OUTPUT, exist_ok=True)
    
    caminho_in = os.path.join(DIRETORIO_INPUT, arquivo_alvo)
    caminho_out = os.path.join(DIRETORIO_OUTPUT, f"extraido_{arquivo_alvo}")
    
    # Valida se o arquivo informado realmente existe na pasta de entrada
    if not os.path.exists(caminho_in):
        print(f"\n❌ Erro: O arquivo '{arquivo_alvo}' não foi encontrado na pasta '{DIRETORIO_INPUT}'.")
        sys.exit(1)

    processar_arquivo_jsonl(caminho_in, caminho_out, arquivo_alvo)

if __name__ == "__main__":
    pipeline_principal()
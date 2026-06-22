import json
import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI, APIError, APIConnectionError, RateLimitError, AuthenticationError
from tqdm import tqdm

# ==========================================
# CONFIGURAÇÕES
# ==========================================
INPUT_FILE = "dados_relatos_relevantes_completos.jsonl"
OUTPUT_FILE = "relatos_rotulados_saude_mental.jsonl"
BATCH_SIZE = 10
MAX_WORKERS = 5
MODEL = "gpt-4o-mini"

client = OpenAI()

file_write_lock = threading.Lock()

SYSTEM_PROMPT = """
You are an expert clinical psychologist and data annotator. Your task is to analyze self-reported text from Reddit posts and classify the cognitive and mental/psychological symptoms present in the text. 

You must evaluate the posts strictly based on the following taxonomy derived from the DSM-5-TR, RDoC, and Neuropsychological Literature.

### COGNITIVE CONSTRUCTS
1. Attention: The ability to sustain focus on relevant information while filtering out distractions; includes selective attention, sustained attention, and divided attention. (Indicators: Difficulty concentrating, easily distracted, trouble completing sustained tasks, mind wandering, inability to multitask).
2. Inhibitory Control: The capacity to suppress prepotent responses, resist distractions, and regulate behavior in accordance with goals. (Indicators: Acting impulsively, difficulty stopping oneself from saying/doing things, trouble waiting turns, interrupting, hasty decisions).
3. Language: The ability to comprehend and produce spoken and written communication. (Indicators: Difficulty finding words, trouble understanding others, problems following complex instructions, reading/writing challenges, reduced verbal fluency).
4. Memory: The encoding, storage, and retrieval of information (episodic and semantic). (Indicators: Forgetting recent events, difficulty remembering names/appointments, losing items, trouble recalling learned info, repeating questions).
5. Working Memory: The temporary storage and manipulation of information necessary for complex cognitive tasks. (Indicators: Difficulty keeping info in mind while using it, trouble following multi-step instructions, losing track mid-task, mental arithmetic challenges, difficulty holding a phone number in mind).

### MENTAL/PSYCHOLOGICAL CONSTRUCTS
1. Affect: The observable expression of emotion. (Indicators: Expressions not matching the situation, rapid emotional shifts, feeling flat/numb, others commenting on expressions, difficulty showing outward emotion).
2. Anxiety: Excessive fear, nervousness, and worry accompanied by physiological arousal. (Indicators: Persistent worry, feeling on edge/restless, racing heart, sweating, trembling, avoidance of triggers, difficulty relaxing/sleeping due to worry, muscle tension).
3. Depression: Persistent low mood and/or anhedonia accompanied by additional symptoms. (Indicators: Feeling sad/empty/hopeless, loss of interest/pleasure, sleep changes, fatigue, worthlessness/guilt, difficulty concentrating, appetite/weight changes, thoughts of death/suicide).
4. Loneliness: The subjective distressing experience of perceived social isolation. (Indicators: Feeling alone around others, lack of close relationships, disconnected/isolated, perceived lack of companionship, feeling left out, no support system).
5. Self-Esteem: The overall subjective evaluation of one's worth or value. (Indicators: Negative self-evaluation, inadequacy/inferiority, lack of confidence, sensitivity to criticism/rejection, unfavorable comparisons, difficulty accepting compliments).
6. Sleep: Disruptions in sleep-wake patterns or sleep quality. (Indicators: Difficulty falling/staying asleep, waking too early, unrefreshed sleep, excessive daytime sleepiness, irregular schedule, non-restorative sleep).
7. Stress: The psychological and physiological response to perceived demands/threats exceeding coping resources. (Indicators: Overwhelmed, under pressure, tension, difficulty managing demands, physical symptoms like headaches/fatigue, irritability related to demands).
8. Well-Being: The presence of positive emotions, life satisfaction, sense of meaning, and positive functioning. (Indicators: Overall life satisfaction, positive emotions, sense of purpose, feeling engaged, positive relationships, personal growth, autonomy).

### CLASSIFICATION RULES:
1. Analyze the provided JSON input containing a list of posts.
2. Determine which symptom(s) from the lists above are explicitly or strongly implicitly reported by the user.
3. You may assign multiple symptoms if applicable. Use the exact names of the constructs provided above.
4. If NONE of the symptoms above fit the post, you must use the label "Other".
5. Explain your reasoning step by step in the "reasoning" field, citing specific words or context from the input text that justify the chosen constructs.
6. Output ONLY a valid JSON object. Do not include conversational text, explanations, or markdown formatting outside of the JSON block.

You will receive a list of posts. You must return a JSON object with a single key "results", containing a list of dictionaries matching the exact output format for each post.

### OUTPUT FORMAT (Inside "results" array):
{"post_id": "<id>", "Symptom": ["<Construct_1>", "<Construct_2>"], "reasoning": "<text>"}
"""

def carregar_ids_processados(caminho_arquivo):
    """Lê o arquivo de saída e retorna um set com os IDs já processados para evitar duplicidade."""
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
                    if "post_id" in post and post["post_id"] not in ids_processados:
                        posts_pendentes.append(post)
                except json.JSONDecodeError:
                    continue
    return posts_pendentes

def criar_lotes(lista, tamanho_lote):
    for i in range(0, len(lista), tamanho_lote):
        yield lista[i:i + tamanho_lote]

def processar_lote_llm(lote):
    lote_para_llm = [
        {
            "post_id": p.get("post_id"),
            "title": p.get("title", ""),
            "body": p.get("body", "")
        }
        for p in lote
    ]
    
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
        return resultados, None

    except APIConnectionError as e:
        return [], f"[ERRO DE CONEXÃO] {e}"
    except RateLimitError as e:
        return [], f"[ERRO DE RATE LIMIT] {e}"
    except AuthenticationError as e:
        return [], f"[ERRO DE AUTENTICAÇÃO] {e}"
    except APIError as e:
        return [], f"[ERRO NA API] HTTP {e.status_code} - {e.message}"
    except json.JSONDecodeError as e:
        return [], f"[ERRO DE PARSING JSON] Falha ao decodificar a resposta da LLM. Detalhes: {e}"
    except Exception as e:
        return [], f"[ERRO DESCONHECIDO] {traceback.format_exc()}"

def processar_dados():
    ids_processados = carregar_ids_processados(OUTPUT_FILE)
    posts_pendentes = carregar_posts_pendentes(INPUT_FILE, ids_processados)

    total_pendentes = len(posts_pendentes)

    print(f"\n--- Resumo Inicial ---")
    print(f"Posts já classificados: {len(ids_processados)}")
    print(f"Posts aguardando processamento: {total_pendentes}")
    print(f"Tamanho do lote: {BATCH_SIZE}")
    print(f"Threads simultâneas: {MAX_WORKERS}")
    print("-" * 22 + "\n")

    if not posts_pendentes:
        print("Nenhum novo post para classificar. Encerrando.")
        return

    lotes = list(criar_lotes(posts_pendentes, BATCH_SIZE))
    
    # Execução Paralela
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as out_file:
        with tqdm(total=total_pendentes, desc="Processando Posts", unit="post") as pbar:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submete todas as tarefas para a pool de threads
                futuros = {executor.submit(processar_lote_llm, lote): lote for lote in lotes}
                
                for futuro in as_completed(futuros):
                    lote_original = futuros[futuro]
                    resultados, erro = futuro.result()
                    
                    if erro:
                        tqdm.write(f"\nFalha ao processar um lote de {len(lote_original)} posts. Erro: {erro}")
                    
                    if resultados:
                        with file_write_lock:
                            for res in resultados:
                                out_file.write(json.dumps(res, ensure_ascii=False) + '\n')
                            out_file.flush()
                    
                    pbar.update(len(lote_original))

if __name__ == "__main__":
    processar_dados()
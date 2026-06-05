import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm 
import re 

DIRETORIO_INPUT = "./dados_reddit"
DIRETORIO_OUTPUT = "./dados_classificados"
MAX_LINHAS_POR_ARQUIVO = 9999999
NUM_CORES_TRABALHADORES = 1
NOME_MODELO = "phi3"

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

PROMPT_SYSTEM = """
Você é um classificador de dados estruturados.
Analise o texto extraído do Reddit e determine se ele é um "Relato" de como o consumo de vídeos curtos (short-form videos) afeta a vida de uma pessoa, ou se é apenas "Ruído" (discussões genéricas, memes, links, dúvidas técnicas, etc).

Retorne APENAS um objeto JSON estrito com as seguintes chaves:
- "e_relato": boolean (true se for um relato pessoal de impacto, false se for ruído)
- "justificativa": string (uma frase curta explicando a decisão)
"""

def classificar_texto(texto_relato):
    try:
        resposta = client.chat.completions.create(
            model=NOME_MODELO,
            messages=[
                {"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user", "content": f"Texto a ser analisado:\n{texto_relato}"}
            ],
            temperature=0.1
        )
        conteudo = resposta.choices[0].message.content
        # tenta extrair JSON mesmo que venha com texto em volta
        match = re.search(r'\{.*?\}', conteudo, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"e_relato": False, "justificativa": "Resposta fora do formato esperado"}
    except Exception as e:
        return {"e_relato": False, "justificativa": f"Erro no processamento: {str(e)}"}

def processar_linha(linha, index, nome_arquivo):
    if not linha.strip():
        return None

    try:
        dado = json.loads(linha)
        texto_title = dado.get("title", "")
        texto_body = dado.get("body", "")

        if not texto_title or not texto_body:
            return None

        texto_relato = texto_title + texto_body

        classificacao = classificar_texto(texto_relato)
        dado["classificacao_triagem"] = classificacao
        
        return dado
    except json.JSONDecodeError:
        tqdm.write(f"[{nome_arquivo}] Erro de sintaxe JSON na linha {index}")
        return None

def processar_arquivo_jsonl(caminho_entrada, caminho_saida, nome_arquivo):
    tqdm.write(f"\nIniciando leitura: {nome_arquivo}")

    # carrega IDs já processados
    ids_processados = set()
    if os.path.exists(caminho_saida):
        with open(caminho_saida, 'r', encoding='utf-8') as f:
            for linha in f:
                if linha.strip():
                    try:
                        dado = json.loads(linha)
                        if "id" in dado:
                            ids_processados.add(dado["id"])
                    except json.JSONDecodeError:
                        pass
        tqdm.write(f"[{nome_arquivo}] {len(ids_processados)} posts já processados, pulando...")

    linhas_para_processar = []

    with open(caminho_entrada, 'r', encoding='utf-8') as infile:
        for index, linha in enumerate(infile):
            if linha.strip():
                try:
                    dado = json.loads(linha)
                    if dado.get("id") in ids_processados:
                        continue  # já foi processado, pula
                    linhas_para_processar.append((linha, index + 1))
                except json.JSONDecodeError:
                    pass

    total_linhas = len(linhas_para_processar)
    if total_linhas == 0:
        tqdm.write(f"[{nome_arquivo}] Nada novo para processar.")
        return

    resultados = [None] * total_linhas

    with ThreadPoolExecutor(max_workers=NUM_CORES_TRABALHADORES) as executor:
        futures = {
            executor.submit(processar_linha, item[0], item[1], nome_arquivo): i
            for i, item in enumerate(linhas_para_processar)
        }

        for future in tqdm(as_completed(futures), total=total_linhas, desc=nome_arquivo, unit="relatos", ncols=100):
            posicao = futures[future]
            resultado_dado = future.result()
            if resultado_dado:
                resultados[posicao] = resultado_dado

    # abre em modo append para não apagar o que já estava
    with open(caminho_saida, 'a', encoding='utf-8') as outfile:
        for res in resultados:
            if res:
                outfile.write(json.dumps(res, ensure_ascii=False) + "\n")

    tqdm.write(f"Concluído: {nome_arquivo} salvo com sucesso.\n")

def pipeline_principal():
    os.makedirs(DIRETORIO_OUTPUT, exist_ok=True)

    if not os.path.exists(DIRETORIO_INPUT):
        print(f"Erro: Diretório '{DIRETORIO_INPUT}' não existe.")
        return

    arquivos = [f for f in os.listdir(DIRETORIO_INPUT) if f.endswith('.jsonl')]

    if not arquivos:
        print(f"Nenhum arquivo encontrado em {DIRETORIO_INPUT}")
        return

    print(f"Processando {len(arquivos)} arquivos com {NUM_CORES_TRABALHADORES} workers paralelos.\n")

    for arquivo in arquivos:
        caminho_in = os.path.join(DIRETORIO_INPUT, arquivo)
        caminho_out = os.path.join(DIRETORIO_OUTPUT, f"classificado_{arquivo}")
        
        processar_arquivo_jsonl(caminho_in, caminho_out, arquivo)

if __name__ == "__main__":
    pipeline_principal()

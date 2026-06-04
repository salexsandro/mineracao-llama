import json
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

prompt_system = """
Você é um especialista em mineração de dados qualitativos.
Analise o relato fornecido e retorne APENAS um objeto JSON estrito com as seguintes chaves:
- "sentimento": string ("Negativo", "Positivo", "Neutro" ou "Misto")
- "impacto_principal": string (Ex: "Privação de Sono", "Ansiedade", "Déficit de Atenção", "Procrastinação", "Entretenimento", "Outro")
- "resumo": string (Um resumo de uma linha do problema relatado)
"""

input_file = "relatos_filtrados_regex.jsonl"
output_file = "relatos_classificados.jsonl"

contagem = 0

def processar_relatos():
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for index, linha in enumerate(infile):
            global contagem
            if  contagem >= 101:
                print("LImite rapaz, ao bolso!!!!")
                break
            if not linha.strip():
                continue
                
            try:
                dado = json.loads(linha)
                
                texto_title = dado.get("title", "") 
                texto_body = dado.get("body", "") 
                if not texto_body or not texto_title:
                    continue

                texto_relato = texto_title + texto_body
                contagem += 1
                print(f"Processando relato de tamanho: {len(texto_relato)} caracteres...")

                resposta = client.chat.completions.create(
                    model="phi3",
                    messages=[
                        {"role": "system", "content": prompt_system},
                        {"role": "user", "content": f"Relato a ser analisado:\n{texto_relato}"}
                    ],
                    response_format={ "type": "json_object" }, 
                    temperature=0.1
                )
                
                conteudo_resposta = resposta.choices[0].message.content

                classificacao_llm = json.loads(conteudo_resposta)
                
                dado["classificacao"] = classificacao_llm
                
                outfile.write(json.dumps(dado, ensure_ascii=False) + "\n")
                
            except json.JSONDecodeError as e:
                print(f"Erro ao parsear o JSON do LLM (ou da linha original): {e}")
            except Exception as e:
                print(f"Erro inesperado: {e}")
            
if __name__ == "__main__":
    print("Iniciando a classificação dos relatos...")
    processar_relatos()
    print(f"Processo finalizado. Dados salvos em {output_file}")

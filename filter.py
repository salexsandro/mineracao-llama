import json

def filtrar_jsonl(arquivo_entrada, arquivo_saida):
    # Contador para acompanharmos quantas linhas foram filtradas
    linhas_filtradas = 0
    
    # Abre o arquivo de entrada para leitura e o de saída para escrita
    with open(arquivo_entrada, 'r', encoding='utf-8') as f_in, \
         open(arquivo_saida, 'w', encoding='utf-8') as f_out:
        
        for linha in f_in:
            # Ignora linhas em branco
            if not linha.strip():
                continue
                
            try:
                # Converte a linha de texto JSON para um dicionário Python
                dados = json.loads(linha)
                
                # Verifica a condição do filtro
                if dados.get("label") == "RELEVANT_REPORT":
                    # Escreve a linha no novo arquivo
                    # Usamos json.dumps para garantir a formatação correta ou gravamos a própria linha
                    f_out.write(json.dumps(dados, ensure_ascii=False) + '\n')
                    linhas_filtradas += 1
            except json.JSONDecodeError:
                print(f"Erro ao decodificar a linha: {linha}")
                
    print(f"Filtragem concluída! {linhas_filtradas} registros foram salvos em '{arquivo_saida}'.")

# Exemplo de uso:
# Substitua 'dados_originais.jsonl' e 'dados_filtrados.jsonl' pelos nomes reais dos seus arquivos
arquivo_input = 'classificacoes.jsonl'
arquivo_output = 'relatos_relevantes.jsonl'

filtrar_jsonl(arquivo_input, arquivo_output)
import json

def cruzar_dados_jsonl(arquivo_filtrados, arquivo_completos, arquivo_saida):
    # Utilizamos um 'set' (conjunto) porque a busca nele é O(1), ou seja, extremamente rápida
    ids_relevantes = set()
    
    # PASSO 1: Extrair todos os IDs do arquivo já filtrado
    with open(arquivo_filtrados, 'r', encoding='utf-8') as f_filtrados:
        for linha in f_filtrados:
            if not linha.strip():
                continue
            try:
                dado = json.loads(linha)
                if "post_id" in dado:
                    ids_relevantes.add(dado["post_id"])
            except json.JSONDecodeError:
                print("Erro ao decodificar linha no arquivo filtrado.")

    print(f"✅ {len(ids_relevantes)} IDs relevantes carregados na memória.")

    # PASSO 2: Buscar os dados completos e salvar no novo arquivo
    linhas_salvas = 0
    with open(arquivo_completos, 'r', encoding='utf-8') as f_completos, \
         open(arquivo_saida, 'w', encoding='utf-8') as f_saida:
        
        for linha in f_completos:
            if not linha.strip():
                continue
            try:
                # Carregamos o JSON apenas para ler o post_id
                dado_completo = json.loads(linha)
                
                # Se o ID bater com os que separamos no passo 1, salvamos a linha
                if dado_completo.get("post_id") in ids_relevantes:
                    # Escrevemos a linha original (em formato string) diretamente no arquivo
                    f_saida.write(linha)
                    linhas_salvas += 1
                    
            except json.JSONDecodeError:
                print("Erro ao decodificar linha no arquivo de dados completos.")
                
    print(f"🎉 Cruzamento concluído! {linhas_salvas} registros salvos em '{arquivo_saida}'.")

# ==========================================
# Configuração dos nomes dos arquivos
# ==========================================
arquivo_filtrados = 'relatos_relevantes.jsonl'
arquivo_completos = 'reddit_clean_data_35k.jsonl'  # O arquivo com o texto completo
arquivo_saida = 'dados_relatos_relevantes_completos.jsonl'

# Executando a função
cruzar_dados_jsonl(arquivo_filtrados, arquivo_completos, arquivo_saida)
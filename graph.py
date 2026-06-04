import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

input_file = "relatos_classificados.jsonl"
output_image = "dashboard_impacto_videos_melhorado.png"

def gerar_visualizacao():
    registros = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for linha in f:
            if not linha.strip():
                continue
            try:
                dado = json.loads(linha)
                if "classificacao" in dado and isinstance(dado["classificacao"], dict):
                    classificacao = dado["classificacao"]
                    registros.append({
                        "Sentimento": classificacao.get("sentimento", "Desconhecido").capitalize(),
                        "Impacto": classificacao.get("impacto_principal", "Outro").title()
                    })
            except Exception as e:
                print(f"Ignorando linha malformada: {e}")

    if not registros:
        print("Nenhum dado válido encontrado para plotar.")
        return

    df = pd.DataFrame(registros)
    df['Sentimento'] = df['Sentimento'].replace({'Misto': 'Neutro/Misto'})
    
    df['Impacto'] = df['Impacto'].apply(lambda x: x[:45] + '...' if len(x) > 45 else x)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 2]})
    fig.suptitle('Análise do Impacto de Short-Form Videos (Reddit)', fontsize=16, fontweight='bold', y=1.05)

    sns.countplot(
        data=df, 
        x='Sentimento', 
        ax=axes[0], 
        palette='viridis',
        order=df['Sentimento'].value_counts().index
    )
    axes[0].set_title('Distribuição Geral de Sentimento', fontsize=14)
    axes[0].set_ylabel('Quantidade de Relatos')
    axes[0].set_xlabel('')
    
    axes[0].tick_params(axis='x', rotation=45)
    
    for p in axes[0].patches:
        axes[0].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='baseline', fontsize=11, color='black', xytext=(0, 5),
                         textcoords='offset points')

    top_10_impactos = df['Impacto'].value_counts().head(10).index
    
    df_top10 = df[df['Impacto'].isin(top_10_impactos)]

    sns.countplot(
        data=df_top10, 
        y='Impacto', 
        ax=axes[1], 
        palette='magma',
        order=top_10_impactos
    )
    axes[1].set_title('Top 10 Principais Áreas Afetadas', fontsize=14)
    axes[1].set_xlabel('Quantidade de Relatos')
    axes[1].set_ylabel('')

    plt.tight_layout()
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f" grafico gerado: {output_image}")

if __name__ == "__main__":
    gerar_visualizacao()

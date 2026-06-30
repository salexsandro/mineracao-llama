"""
Combina posts_with_topics.jsonl (tópicos do BERTopic) com
relatos_rotulados_saude_mental.jsonl (sintomas classificados via LLM).

Junção feita por post_id. Posts sem correspondência em um dos dois
arquivos são reportados no log e excluídos do output final.
"""

import json
import os
import sys
import io
import logging

# ─────────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────────

TOPICS_FILE   = "posts_with_topics.jsonl"
SYMPTOMS_FILE = "relatos_rotulados_saude_mental.jsonl"
OUTPUT_FILE   = "posts_topics_symptoms_merged.jsonl"

# ─────────────────────────────────────────────
# LOGGING — UTF-8 explícito (compatibilidade Windows)
# ─────────────────────────────────────────────

_stream_handler = logging.StreamHandler(
    stream=io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[_stream_handler]
)


def load_jsonl(path: str) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logging.warning(f"  Linha {line_num} inválida em {path}: {e}")
    return records


def merge(topics_path: str, symptoms_path: str, output_path: str):
    topics_data   = load_jsonl(topics_path)
    symptoms_data = load_jsonl(symptoms_path)

    logging.info(f"Tópicos:  {len(topics_data)} registros lidos de {topics_path}")
    logging.info(f"Sintomas: {len(symptoms_data)} registros lidos de {symptoms_path}")

    # Indexa sintomas por post_id para junção O(1)
    symptoms_by_id = {s["post_id"]: s for s in symptoms_data if "post_id" in s}

    merged       = []
    missing_symp = []   # posts com tópico mas sem classificação de sintoma
    missing_meta = []   # sintomas sem correspondência no arquivo de tópicos (apenas para log)

    topic_ids_seen = set()

    for t in topics_data:
        post_id = t.get("post_id")
        topic_ids_seen.add(post_id)

        symptom_entry = symptoms_by_id.get(post_id)

        if symptom_entry is None:
            missing_symp.append(post_id)
            continue

        merged.append({
            "post_id":      post_id,
            "title":        t.get("title"),
            "Symptom":      symptom_entry.get("Symptom", []),
            "reasoning":    symptom_entry.get("reasoning"),
            "topic_id":     t.get("topic_id"),
            "topic_label":  t.get("topic_label"),
            "confidence":   t.get("confidence"),
        })

    # Sintomas que não bateram com nenhum post do arquivo de tópicos
    for post_id in symptoms_by_id:
        if post_id not in topic_ids_seen:
            missing_meta.append(post_id)

    with open(output_path, "w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logging.info(f"\n{'='*60}")
    logging.info(f"Registros combinados com sucesso : {len(merged)}")
    logging.info(f"Posts sem classificação de sintoma: {len(missing_symp)}")
    logging.info(f"Sintomas sem post correspondente   : {len(missing_meta)}")
    logging.info(f"{'='*60}")
    logging.info(f"Arquivo salvo: {output_path}")

    if missing_symp:
        sample = missing_symp[:5]
        logging.info(f"  Exemplos sem sintoma: {sample}")
    if missing_meta:
        sample = missing_meta[:5]
        logging.info(f"  Exemplos sem post:    {sample}")

    return merged


if __name__ == "__main__":
    topics_arg   = sys.argv[1] if len(sys.argv) > 1 else TOPICS_FILE
    symptoms_arg = sys.argv[2] if len(sys.argv) > 2 else SYMPTOMS_FILE
    output_arg   = sys.argv[3] if len(sys.argv) > 3 else OUTPUT_FILE

    merge(topics_arg, symptoms_arg, output_arg)

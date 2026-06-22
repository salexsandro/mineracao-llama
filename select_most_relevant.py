import json
from pathlib import Path
import numpy as np
import pandas as pd

# =====================
# CONFIG
# =====================

INPUT_FILE = "grid_results_2/15c_20n/posts_with_topics.jsonl"
OUTPUT_DIR = Path("topics_most_relevant")

TOPICS = list(range(10))  # 0-13
N_POSTS = 50

# =====================
# LOAD DATA
# =====================

records = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

df = pd.DataFrame(records)

# =====================
# OUTPUT FOLDER
# =====================

OUTPUT_DIR.mkdir(exist_ok=True)

all_selected = []

# =====================
# SELECT TOP POSTS
# =====================

for topic_id in TOPICS:

    topic_df = (
        df[df["topic_id"] == topic_id]
        .sort_values("confidence", ascending=False)
        .head(N_POSTS)
        .copy()
    )

    topic_df["rank_in_topic"] = range(1, len(topic_df) + 1)

    all_selected.append(topic_df)

    output_file = OUTPUT_DIR / f"topic_{topic_id:02d}.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        for _, row in topic_df.iterrows():

            record = {
                "rank_in_topic": int(row["rank_in_topic"]),
                "topic_id": int(row["topic_id"]),
                "topic_label": row["topic_label"],
                "confidence": float(row["confidence"]),
                "post_id": row["post_id"],
                "title": row["title"],
                "body": row["body"],
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"Topic {topic_id}: "
        f"{len(topic_df)} posts saved -> {output_file}"
    )

# =====================
# COMBINED FILE
# =====================

combined_df = pd.concat(all_selected)

combined_df = combined_df.sort_values(
    ["topic_id", "rank_in_topic"]
)

combined_file = OUTPUT_DIR / "all_topics_top50.jsonl"

with open(combined_file, "w", encoding="utf-8") as f:

    for _, row in combined_df.iterrows():

        record = {
            "rank_in_topic": int(row["rank_in_topic"]),
            "topic_id": int(row["topic_id"]),
            "topic_label": row["topic_label"],
            "confidence": float(row["confidence"]),
            "post_id": row["post_id"],
            "title": row["title"],
            "body": row["body"],
        }

        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"\nCombined file saved: {combined_file}")
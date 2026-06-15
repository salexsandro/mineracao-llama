import json

CLASSIFICATIONS_FILE = "classificado_reddit_clean_data_35k_slim.jsonl"
DATA_FILE = "reddit_clean_data_35k_slim.jsonl"
OUTPUT_FILE = "reddit_relevant_only.jsonl"

# Load all post_ids classified as RELEVANT_REPORT
relevant_ids = set()
with open(CLASSIFICATIONS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        if entry.get("label") == "RELEVANT_REPORT":
            relevant_ids.add(entry["post_id"])

print(f"Found {len(relevant_ids)} RELEVANT_REPORT post IDs.")

# Filter the data file, keeping only posts in relevant_ids
kept = 0
removed = 0
with open(DATA_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
    for line in infile:
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        if entry.get("post_id") in relevant_ids:
            outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")
            kept += 1
        else:
            removed += 1

print(f"Done. Kept: {kept} | Removed (unclassified or non-relevant): {removed}")
print(f"Output written to: {OUTPUT_FILE}")

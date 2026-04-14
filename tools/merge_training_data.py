"""Merge multiple training data JSON files into one."""
import json
from pathlib import Path

data_dir = Path(__file__).parent.parent / "data"
output = data_dir / "training_data.json"

all_data = []
for f in sorted(data_dir.glob("training_data*.json")):
    with open(f) as fh:
        items = json.load(fh)
        print(f"{f.name}: {len(items)} examples")
        all_data.extend(items)

# Deduplicate by source_id
seen = set()
deduped = []
for item in all_data:
    sid = item.get("source_id", id(item))
    if sid not in seen:
        seen.add(sid)
        deduped.append(item)

with open(output, "w") as f:
    json.dump(deduped, f, indent=2, ensure_ascii=False)

print(f"\nTotal: {len(deduped)} unique examples -> {output}")

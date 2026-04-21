import json
import re

with open("results/debug_stage1_raw.json") as f:
    text = f.read()

# Test extraction
text = text.strip()
text = re.sub(r"^```(?:json)?\s*", "", text)
text = re.sub(r"\s*```$", "", text)
try:
    data = json.loads(text)
    print("Direct parse: SUCCESS")
    print(f"Got {len(data)} tasks")
except json.JSONDecodeError as e:
    print(f"Direct parse failed: {e}")

# Test regex extraction
match = re.search(r"\[.*\]", text, re.DOTALL)
if match:
    try:
        data2 = json.loads(match.group(0))
        print("Regex match: SUCCESS")
        print(f"Got {len(data2)} tasks")
    except json.JSONDecodeError as e:
        print(f"Regex failed: {e}")
else:
    print("No regex match found")

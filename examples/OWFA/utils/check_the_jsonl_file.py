import json

jsonl_file_path = "../data/wiki/olmo-mix-1124-train-00000-of-00036.jsonl"  # Replace with your actual file path

with open(jsonl_file_path, 'r', encoding='utf-8') as f:
    count = 0
    for line in f:
        if count >= 5:
            break
        data = json.loads(line)
        print(data)
        count += 1

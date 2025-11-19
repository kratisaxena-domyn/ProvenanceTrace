
import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer

# Paths
# data_dir = "../data/wiki"
# output_dir = "../data/wiki_unigram_probs"
# os.makedirs(output_dir, exist_ok=True)
# output_path = os.path.join(output_dir, "olmo-7b_wiki_unigram_probs.json")

def calculate_and_save_unigram_probability(data_dir: str, output_path: str, tokenizer: AutoTokenizer):
	"""
	Calculate unigram probabilities from a directory of Wikipedia JSONL files
	and save the probabilities as a JSON file.

	Args:
		data_dir (str): Directory containing Wikipedia JSONL files.
		output_path (str): Path to save the unigram probabilities JSON file.
	"""
	# Ensure output directory exists
	os.makedirs(os.path.dirname(output_path), exist_ok=True)

	# Load tokenizer
	# tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B", add_bos_token=False, add_eos_token=False, trust_remote_code=True)
	vocab_size = tokenizer.vocab_size

	# Count tokens
	token_counts = [0] * vocab_size
	total_tokens = 0

	for fname in tqdm(sorted(os.listdir(data_dir)), desc="Processing wiki JSONL files"):
		if not fname.endswith(".jsonl"): continue
		fpath = os.path.join(data_dir, fname)
		with open(fpath, 'r', encoding='utf-8') as f:
			for line in f:
				data = json.loads(line)
				text = data.get('text', '')
				token_ids = tokenizer.encode(text, add_special_tokens=False)
				for tid in token_ids:
					if 0 <= tid < vocab_size:
						token_counts[tid] += 1
						total_tokens += 1

	# Compute probabilities
	token_probs = {str(tid): count / total_tokens if total_tokens > 0 else 0.0 for tid, count in enumerate(token_counts)}

	# Save as JSON
	with open(output_path, 'w', encoding='utf-8') as f:
		json.dump(token_probs, f)

	print(f"Saved unigram probabilities for {vocab_size} tokens to {output_path}")

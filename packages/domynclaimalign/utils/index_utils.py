from infini_gram.engine import InfiniGramEngine
import json

def load_engine_util(tokenizer, data_index_dir):
    return InfiniGramEngine(index_dir=data_index_dir, eos_token_id=tokenizer.eos_token_id)


def load_unigram_probs(UNIGRAM_PATH):
    with open(UNIGRAM_PATH, 'r', encoding='utf-8') as f:
        loaded_probs = json.load(f)
    return {int(tid): float(prob) for tid, prob in loaded_probs.items()}

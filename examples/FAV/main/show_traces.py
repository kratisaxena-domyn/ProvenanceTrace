import sys
sys.path.append('..')
import os
import json
from icecream import ic

from domynclaimalign.utils.other_utils import extract_entities_keywords

def extract_turn_trace(turn_number, AGENT_LOG_DIR='../agent_logs', AGENT_LOG_FILES=None):
	traces = { 'answer': None, 'steps': [] }
	for log_file in AGENT_LOG_FILES:
		file_path = os.path.join(AGENT_LOG_DIR, log_file)
		if not os.path.exists(file_path):
			continue
		with open(file_path, 'r') as f:
			for line in f:
				try:
					entry = json.loads(line)
				except Exception:
					continue
				if entry.get('turn') == turn_number:
					step = entry.get('step')
					if log_file == 'orchestrator_log.jsonl':
						if step == 'finalize':
							traces['answer'] = entry.get('data')
					else:
						traces['steps'].append(entry.get('data'))
						traces['steps'].append(entry.get('thought'))
	return traces

def get_entities_keywords_from_answer(answer):
    if not answer:
        return {'entities': [], 'keywords': []}
    return extract_entities_keywords(answer)

def remove_duplicates(items):
	seen = set()
	result = []
	for item in items:
		# Convert dicts (or other unhashables) to a frozenset of key-value pairs for comparison
		key = item if isinstance(item, (str, int, float, tuple)) else str(item)
		if key not in seen:
			seen.add(key)
			result.append(item)
	return result

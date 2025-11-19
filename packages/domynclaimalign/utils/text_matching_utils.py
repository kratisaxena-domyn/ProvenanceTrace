
from icecream import ic
from concurrent.futures import ThreadPoolExecutor

def merge_overlapping_spans(spans):
    if not spans:
        return []
    # Assume spans are sorted by start
    merged = []
    last = None
    for span in spans:
        if last is None:
            last = dict(span)
        elif span['start'] <= last['end']:
            last['end'] = max(last['end'], span['end'])
            last['length'] = last['end'] - last['start']
        else:
            merged.append(last)
            last = dict(span)
    if last is not None:
        merged.append(last)
    return merged

def match_full_phrases_in_index(engine, phrases, tokenizer, period_token_id, newline_token_id):
    results = []
    for phrase in phrases:
        phrase = phrase.strip()
        token_ids = tokenizer.encode(phrase)
        if not token_ids:
            continue
        result = engine.find(token_ids)
        segs = result.get('segment_by_shard', [])
        if segs and any(int(seg[1]) - int(seg[0]) > 0 for seg in segs):
            span = {
                'start': 0,
                'end': len(token_ids),
                'length': len(token_ids),
                'sa_start': [int(seg[0]) for seg in segs] if segs else [0]*36,
                'sa_end': [int(seg[1]) for seg in segs] if segs else [0]*36,
                'phrase': phrase
            }
            results.append(span)

    ic(len(results))
    return results



def is_begin_of_word(idx, tokenizer, decoded_tokens):
    if tokenizer is None or idx == 0:
        return True
    return decoded_tokens[idx].startswith(' ') or idx == 0

def contains_delimiter(tokens, period_token_id=None, newline_token_id=None):
    if len(tokens) <= 1:
        return False
    return any(t in (period_token_id, newline_token_id) for t in tokens[:-1])

def ends_on_word_boundary(tokens, tokenizer):
    if tokenizer is None or not tokens:
        return True
    decoded = tokenizer.decode(tokens, skip_special_tokens=True)
    reencoded = tokenizer.encode(decoded, add_special_tokens=False)
    return list(tokens) == reencoded

def get_longest_prefix_len(suffix, engine):
    result = engine.find(suffix)
    segs = result.get('segment_by_shard', [])
    max_len = 0
    for seg in segs:
        sa_start, sa_end = map(int, seg)
        length = sa_end - sa_start
        if length > max_len:
            max_len = length
    if max_len > 0:
        return max_len
    for l in range(len(suffix)-1, 0, -1):
        result = engine.find(suffix[:l])
        segs = result.get('segment_by_shard', [])
        found = any((int(seg[1]) - int(seg[0])) > 0 for seg in segs)
        if found:
            return l
    return 0

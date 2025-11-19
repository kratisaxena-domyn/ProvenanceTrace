from domynclaimalign.utils.text_matching_utils import is_begin_of_word, get_longest_prefix_len, contains_delimiter, ends_on_word_boundary
from concurrent.futures import ThreadPoolExecutor

# =========================
# Maximal span matching (optimized)
# =========================
def maximal_matching_spans(engine, input_ids, period_token_id, newline_token_id, tokenizer=None):
    n = len(input_ids)
    spans = []
    # Precompute decoded tokens for word boundary checks
    decoded_tokens = None
    if tokenizer is not None:
        decoded_tokens = [tokenizer.decode([t], skip_special_tokens=True) for t in input_ids]

    # Avoid thread pool overhead for short answers, use parallel for long
    if n < 32:
        for b in range(n):
            if not is_begin_of_word(b, tokenizer, decoded_tokens):
                continue
            suffix = input_ids[b:]
            length = get_longest_prefix_len(suffix, engine)
            if length == 0:
                continue
            while length > 0:
                tokens = input_ids[b:b+length]
                if contains_delimiter(tokens, period_token_id, newline_token_id) or not ends_on_word_boundary(tokens, tokenizer):
                    length -= 1
                else:
                    break
            if length == 0:
                continue
            spans.append((b, b+length))
    else:
        # Use ThreadPoolExecutor for large n, but chunk to avoid too many threads
        def process_start(b):
            if not is_begin_of_word(b, tokenizer, decoded_tokens):
                return None
            suffix = input_ids[b:]
            length = get_longest_prefix_len(suffix, engine)
            if length == 0:
                return None
            while length > 0:
                tokens = input_ids[b:b+length]
                if contains_delimiter(tokens, period_token_id, newline_token_id) or not ends_on_word_boundary(tokens, tokenizer):
                    length -= 1
                else:
                    break
            if length == 0:
                return None
            return (b, b+length)
        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(executor.map(process_start, range(n)))
        spans = [res for res in results if res is not None]

    # Merge overlapping spans in a single pass (sorted by start)
    spans = sorted(spans, key=lambda x: x[0])
    maximal_spans = []
    max_end = -1
    for b, e in spans:
        if e > max_end:
            maximal_spans.append({"start": b, "end": e, "length": e-b})
            max_end = e

    # Batch engine.find for all maximal spans for sa_start/sa_end
    for span in maximal_spans:
        tokens = input_ids[span['start']:span['end']]
        result = engine.find(tokens)
        segs = result.get('segment_by_shard', [])
        span['sa_start'] = [int(seg[0]) for seg in segs] if segs else [0]*36
        span['sa_end'] = [int(seg[1]) for seg in segs] if segs else [0]*36
    return maximal_spans

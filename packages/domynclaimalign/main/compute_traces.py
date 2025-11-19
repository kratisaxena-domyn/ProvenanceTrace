import time, math, json
from rank_bm25 import BM25Okapi
from keybert import KeyBERT
from icecream import ic

from domynclaimalign.utils.model_utils import load_model_and_tokenizer, load_spacy
from domynclaimalign.utils.index_utils import load_engine_util as load_engine
from domynclaimalign.utils.index_utils import load_unigram_probs
from domynclaimalign.utils.other_utils import compute_span_unigram_prob
from domynclaimalign.utils.text_matching_utils import merge_overlapping_spans
from domynclaimalign.utils.json_utils import get_jsonl_store
from domynclaimalign.main.match_entities_in_text import optimized_entity_doc_processing
from domynclaimalign.main.maximal_match_text import maximal_matching_spans

# =========================
# Traces computation
# =========================
def compute_traces(query, answer, MODEL_NAME, MODEL_CACHE, UNIGRAM_PATH, SPACY_MODEL, MODEL_ID_sentence_transformer, BASE_DIR, DATA_INDEX_DIR):
    timings = {}

    # Load resources
    t0 = time.perf_counter()
    tokenizer, _model, __build_class__ = load_model_and_tokenizer(MODEL_NAME, MODEL_CACHE)
    engine = load_engine(tokenizer, DATA_INDEX_DIR)
    token_unigram_probs = load_unigram_probs(UNIGRAM_PATH)
    nlp = load_spacy(SPACY_MODEL)
    timings['loading_resources'] = time.perf_counter() - t0

    # Entities/keywords from both query and answer; fallback to KeyBERT if neither has entities
    t0 = time.perf_counter()
    doc_query = nlp(query)
    doc_answer = nlp(answer)
    entities_query = [(ent.text, ent.label_) for ent in doc_query.ents]
    entities_answer = [(ent.text, ent.label_) for ent in doc_answer.ents]
    entities = list({(e[0], e[1]) for e in entities_query + entities_answer})

    if not entities:
        try:
            
            kw_model = KeyBERT()
            keywords_query = kw_model.extract_keywords(query, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=1)
            keywords_answer = kw_model.extract_keywords(answer, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=1)
            # Combine and deduplicate
            keywords = list({kw for kw in keywords_query + keywords_answer if isinstance(kw, str) and kw.strip()})
            if not keywords:
                # If KeyBERT returns tuples (kw, score)
                keywords = list({kw[0] for kw in keywords_query + keywords_answer if isinstance(kw, (list, tuple)) and len(kw) > 0 and isinstance(kw[0], str) and kw[0].strip()})
            entities = [(kw, 'KEYPHRASE') for kw in keywords]

        except ImportError:
            return None, None, None, "KeyBERT not installed. Please install with 'pip install keybert'."
    ic(entities)
    timings['entity_extraction'] = time.perf_counter() - t0

    # Token IDs and special token ids
    t0 = time.perf_counter()
    answer_token_ids = tokenizer.encode(answer, add_special_tokens=False)
    period_token_id = tokenizer.convert_tokens_to_ids('.')
    newline_token_id = tokenizer.convert_tokens_to_ids('\n')
    timings['tokenize_answer'] = time.perf_counter() - t0

    # Optimized entity-based retrieval + scoring
    restricted_docs_query, restricted_docs_answer = optimized_entity_doc_processing(
        engine, entities, entities_query, tokenizer, period_token_id, newline_token_id, nlp, answer, timings, MODEL_ID_sentence_transformer, BASE_DIR
    )

    # General maximal span matching
    t0 = time.perf_counter()

    # Increase max_workers for parallelism if supported by maximal_matching_spans
    maximal_spans = maximal_matching_spans(
        engine, answer_token_ids, period_token_id, newline_token_id, tokenizer=tokenizer
    )
    L = len(answer_token_ids)
    K = max(1, int(math.ceil(0.05 * L)))

    for span in maximal_spans:
        span_token_ids = answer_token_ids[span['start']:span['end']]
        span['unigram_prob'] = compute_span_unigram_prob(span_token_ids, token_unigram_probs)

    filtered_spans = sorted(maximal_spans, key=lambda s: s['unigram_prob'])[:K]
    merged_spans = merge_overlapping_spans(filtered_spans)
    timings['maximal_span_matching'] = time.perf_counter() - t0


    # Optimized: reuse JsonlStore, batch file reads, and use dict for merging
    t0 = time.perf_counter()
    total_docs = []
    doc_keys = set()
    store = get_jsonl_store(BASE_DIR)  # Reuse store instance
    # Collect all doc requests for batch file reading
    doc_requests = []  # (jsonl_path, line_num, doc, span_text, key)
    for span in merged_spans:
        span_text = tokenizer.decode(answer_token_ids[span['start']:span['end']], skip_special_tokens=True)
        shard_indices = range(len(span['sa_start'])) if isinstance(span['sa_start'], list) else [0]
        for shard_idx in shard_indices:
            shard_start = span['sa_start'][shard_idx] if isinstance(span['sa_start'], list) else span['sa_start']
            shard_end = span['sa_end'][shard_idx] if isinstance(span['sa_end'], list) else span['sa_end']
            for rank in range(shard_start, shard_end):
                doc = engine.get_doc_by_rank(s=shard_idx, rank=rank)
                meta_raw = doc.get('metadata', '{}')
                try:
                    meta = json.loads(meta_raw)
                except Exception:
                    meta = {}
                jsonl_path = meta.get('path')
                line_num = meta.get('linenum')
                key = (doc.get('doc_ix'), meta.get('path', ''))
                if jsonl_path is None or line_num is None or key in doc_keys:
                    continue
                doc_requests.append((jsonl_path, int(line_num), doc, span_text, key))
                doc_keys.add(key)

    # Batch file reads: group by jsonl_path
    from collections import defaultdict
    path_to_requests = defaultdict(list)
    for jsonl_path, line_num, doc, span_text, key in doc_requests:
        path_to_requests[jsonl_path].append((line_num, doc, span_text, key))

    # Read all needed lines per file
    key_to_docinfo = {}  # key -> doc info dict
    for jsonl_path, reqs in path_to_requests.items():
        # Optionally, sort by line_num for efficiency
        reqs_sorted = sorted(reqs, key=lambda x: x[0])
        for line_num, doc, span_text, key in reqs_sorted:
            rec = store.get_line_json(jsonl_path, line_num)
            if not rec or "text" not in rec:
                continue
            full_doc_text = rec["text"]
            # Use dict for merging
            if key not in key_to_docinfo:
                key_to_docinfo[key] = {
                    "text": span_text,
                    "doc": doc,
                    "span_texts": [span_text],
                    "full_doc_text": full_doc_text
                }
            else:
                key_to_docinfo[key]["text"] += " ... " + span_text
                key_to_docinfo[key]["span_texts"].append(span_text)
    total_docs = list(key_to_docinfo.values())
    timings['document_retrieval_merging'] = time.perf_counter() - t0

    # Provenance ranking: BM25 on combined query + answer
    t0 = time.perf_counter()
    doc_texts = [d["text"] for d in total_docs]
    bm25_query = f"{query} {answer}"
    # Use spaCy for tokenization for better accuracy
    def spacy_tokenize(text):
        return [t.text for t in nlp(text)]
    tokenized_corpus = [spacy_tokenize(doc) for doc in doc_texts]
    tokenized_query = spacy_tokenize(bm25_query)
    if tokenized_corpus:
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(tokenized_query)
        for i, doc in enumerate(total_docs):
            doc["bm25_score"] = float(bm25_scores[i])
        sorted_docs = sorted(total_docs, key=lambda d: -d.get("bm25_score", 0.0))
    else:
        sorted_docs = total_docs
    timings['bm25_scoring_sorting'] = time.perf_counter() - t0

    ic(timings)
    return restricted_docs_query, restricted_docs_answer, sorted_docs, None, timings, entities

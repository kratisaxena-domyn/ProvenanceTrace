import time, json, os, re
from icecream import ic
import ahocorasick  # pyahocorasick [1]
from concurrent.futures import ThreadPoolExecutor
from domynclaimalign.utils.json_utils import get_jsonl_store
from domynclaimalign.utils.text_matching_utils import match_full_phrases_in_index
from domynclaimalign.utils.other_utils import _norm, _contains_span, sbert_sim_batch, jaccard, seqmatch
from domynclaimalign.utils.model_utils import load_sentence_transformer

# =========================
# Optimized entity-based doc processing
# =========================
def optimized_entity_doc_processing(engine, entities_all, entities_query, tokenizer, period_token_id, newline_token_id, nlp, answer, timings, MODEL_ID_sentence_transformer, data_base_dir):
    t0 = time.perf_counter()
    # Pre-create the store outside the function to avoid repeated initialization
    store = get_jsonl_store(data_base_dir)
    entity_docs = []            # global list of collected docs (across entities)
    entity_doc_keys = set()     # global dedupe key set
    restricted_docs_query, restricted_docs_answer = [], []

    IGNORE_LABELS = {'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL'}

    # Helper: robust substring check (optional)
    
    for entity in entities_all:

        ic("in entity matching", entity)
        # Each loop is a single (text, label) entity tuple
        if not isinstance(entity, (list, tuple)) or len(entity) < 2:
            continue
        ent_text = entity[0].strip()
        ent_label = entity[1].strip().upper()

        # Skip ignored labels early
        if not ent_text or ent_label in IGNORE_LABELS:
            continue

        # 1) Find spans for JUST this entity phrase
        phrases = [ent_text]  # do not iterate strings; use the full text
        entity_spans = match_full_phrases_in_index(
            engine, phrases, tokenizer, period_token_id, newline_token_id
        )

        MAX_DOCS_PER_SPAN = 10
        MAX_ENTITY_DOCS = 2000

        # Track where new docs begin in the global list for this entity
        start_len = len(entity_docs)
        ic(start_len)
        # 2) Flatten shards and collect candidate docs
        flattened = []
        for span in entity_spans:
            sa_starts = span['sa_start'] if isinstance(span['sa_start'], list) else [span['sa_start']]
            sa_ends   = span['sa_end']   if isinstance(span['sa_end'],   list) else [span['sa_end']]
            phrase = span['phrase']
            for shard_idx, (st, en) in enumerate(zip(sa_starts, sa_ends)):
                flattened.append((shard_idx, int(st), int(en), phrase))


        for shard_idx, shard_start, shard_end, phrase in flattened:
            if len(entity_doc_keys) >= MAX_ENTITY_DOCS:
                break
            limit_end = min(shard_end, shard_start + MAX_DOCS_PER_SPAN)
            for rank in range(shard_start, limit_end):
                if len(entity_doc_keys) >= MAX_ENTITY_DOCS:
                    break
                doc = engine.get_doc_by_rank(s=shard_idx, rank=rank)  # rank->doc via index [4]
                meta_raw = doc.get('metadata', '{}')
                try:
                    meta = json.loads(meta_raw)
                except Exception:
                    meta = {}
                key = (doc.get('doc_ix'), meta.get('path', ''))
                if key not in entity_doc_keys:
                    entity_doc_keys.add(key)
                    # store span_text for this entity; text will be loaded below
                    entity_docs.append({"doc": doc, "span_text": phrase})
        

        # 3) Load text ONLY for docs collected in this entity iteration
        if len(entity_docs) == start_len:
            # nothing added for this entity
            continue
        new_slice = slice(start_len, len(entity_docs))
        new_docs = entity_docs[new_slice]

        
        
        
        # Group documents by file path for batch processing
        docs_by_path = {}
        for i, d in enumerate(new_docs):
            doc = d['doc']
            meta_raw = doc.get('metadata', '{}')
            try:
                meta = json.loads(meta_raw)
            except Exception:
                continue
            
            jsonl_path = meta.get('path')
            line_num = meta.get('linenum')
            if jsonl_path is None or line_num is None:
                continue
                
            if jsonl_path not in docs_by_path:
                docs_by_path[jsonl_path] = []
            docs_by_path[jsonl_path].append((i, int(line_num)))
        
        # Process files in batches
        texts = [None] * len(new_docs)
        
        def process_file_batch(path_and_docs):
            path, doc_indices = path_and_docs
            file_texts = {}
            try:
                # Sort by line number for potential I/O efficiency
                sorted_docs = sorted(doc_indices, key=lambda x: x[1])
                for doc_idx, line_num in sorted_docs:
                    rec = store.get_line_json(path, line_num)
                    file_texts[doc_idx] = rec.get('text') if rec else None
            except Exception as e:
                ic(f"Error processing file {path}: {e}")
                for doc_idx, _ in doc_indices:
                    file_texts[doc_idx] = None
            return file_texts
        
        # Use ThreadPoolExecutor on file batches instead of individual docs
        max_workers = min(16, len(docs_by_path), (os.cpu_count() or 4))
        if max_workers > 1 and len(docs_by_path) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                file_results = ex.map(process_file_batch, docs_by_path.items())
                for file_texts in file_results:
                    for doc_idx, text in file_texts.items():
                        texts[doc_idx] = text
        else:
            # Single-threaded for small workloads
            for path_and_docs in docs_by_path.items():
                file_texts = process_file_batch(path_and_docs)
                for doc_idx, text in file_texts.items():
                    texts[doc_idx] = text

        # Write back texts and filter out any that failed to load
        filtered_new_docs = []
        for d, txt in zip(new_docs, texts):
            if not txt:
                continue
            d['text'] = txt
            filtered_new_docs.append(d)
        # Replace the slice with only loaded docs
        entity_docs[new_slice] = filtered_new_docs

        # 4) Fast multi-phrase scan for THIS entity only (case-insensitive)
        # Build automaton with lowercased pattern, payload as original text

        unique_phrases = [ent_text]  # only the full entity phrase, not its first char
        unique_phrases_norm = [_norm(p) for p in unique_phrases]

        matched_flags = [False] * len(filtered_new_docs)
        per_doc_matches = [[] for _ in range(len(filtered_new_docs))]
        ic(unique_phrases)

        try:
              # pyahocorasick [1]
            A = ahocorasick.Automaton()
            for p_norm, p in zip(unique_phrases_norm, unique_phrases):
                A.add_word(p_norm, p)  # payload keeps original phrase
            A.make_automaton()
            for i, d in enumerate(filtered_new_docs):
                text_norm = _norm(d['text'])
                for _, hit in A.iter(text_norm):
                    matched_flags[i] = True
                    per_doc_matches[i].append(hit)
        except Exception:
            # Regex fallback: case-insensitive, word-boundary aware
            pats = [re.compile(r'\b' + re.escape(_norm(p)) + r'\b', flags=re.IGNORECASE) for p in unique_phrases]
            for i, d in enumerate(filtered_new_docs):
                hits = []
                text_norm = _norm(d['text'])
                for p, pat in zip(unique_phrases, pats):
                    if pat.search(text_norm):
                        hits.append(p)
                if hits:
                    matched_flags[i] = True
                    per_doc_matches[i] = hits

        # Optional: strict verify substring presence with normalization to avoid false positives
        for i, d in enumerate(filtered_new_docs):
            if matched_flags[i]:
                # ensure the real phrase is contained after normalization
                if not any(_contains_span(d['text'], h) for h in per_doc_matches[i]):
                    matched_flags[i] = False
                    per_doc_matches[i] = []

        kept_indices = [i for i, flag in enumerate(matched_flags) if flag]
        if not kept_indices:
            continue

        # Route to query vs answer collections
        if entity in entities_query:
            restricted_docs_query.extend(
                {"doc": d, "matched_spans": sorted(set(per_doc_matches[i])), "entities": []}
                for i, d in enumerate(filtered_new_docs) if i in kept_indices
            )
        else:
            restricted_docs_answer.extend(
                {"doc": d, "matched_spans": sorted(set(per_doc_matches[i])), "entities": []}
                for i, d in enumerate(filtered_new_docs) if i in kept_indices
            )

        ic(len(restricted_docs_query), len(restricted_docs_answer))
        
    timings['entity_based_doc_filtering'] = time.perf_counter() - t0

    # Process answer entities once
    lm_ents = set(str(e.text) for e in nlp(answer).ents)
    
    # Load model once
    sbert_model = load_sentence_transformer(MODEL_ID_sentence_transformer)
    
    # Combine all docs for batch processing
    all_docs = restricted_docs_query + restricted_docs_answer
    if not all_docs:
        timings['alignment_consistency_scoring'] = time.perf_counter() - t0
        ic("Entities done")
        return restricted_docs_query, restricted_docs_answer
    
    # Deduplicate texts for SBERT processing
    unique_texts = []
    text_to_indices = {}
    doc_texts = []
    
    for i, d in enumerate(all_docs):
        text = d['doc']['text']
        doc_texts.append(text)
        if text not in text_to_indices:
            text_to_indices[text] = []
            unique_texts.append(text)
        text_to_indices[text].append(i)
    
    # Batch SBERT on unique texts only
    unique_sbert_sims = sbert_sim_batch(sbert_model, answer, unique_texts, normalize=True)
    
    # Map similarities back to all documents
    sbert_sims = [0.0] * len(all_docs)
    for unique_idx, (text, sim) in enumerate(zip(unique_texts, unique_sbert_sims)):
        for doc_idx in text_to_indices[text]:
            sbert_sims[doc_idx] = sim
    
    # Precompute entity sets and similarities
    doc_entity_sets = []
    jaccard_scores = []
    desc_sims = []
    
    for d in all_docs:
        doc_ents = set(str(e) for e in d['entities'])
        doc_entity_sets.append(doc_ents)
        
        # Jaccard similarity
        j = jaccard(lm_ents, doc_ents)
        jaccard_scores.append(j)
        
        # Description similarity
        if lm_ents and doc_ents:
            desc_sim = max((seqmatch(le, de) for le in lm_ents for de in doc_ents), default=0.0)
        else:
            desc_sim = 0.0
        desc_sims.append(desc_sim)
    
    # Compute final alignment scores
    for i, (d, j, desc_sim, sbert_sim) in enumerate(zip(all_docs, jaccard_scores, desc_sims, sbert_sims)):
        d['alignment_score'] = 0.5 * j + 0.25 * desc_sim + 0.25 * float(sbert_sim)

    timings['alignment_consistency_scoring'] = time.perf_counter() - t0

    ic("Entities done")
    return restricted_docs_query, restricted_docs_answer

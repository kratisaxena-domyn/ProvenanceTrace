from difflib import SequenceMatcher
import numpy as np
import unicodedata, re
from sentence_transformers import util
import spacy


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def seqmatch(a, b):
    return SequenceMatcher(None, a, b).ratio()


def compute_span_unigram_prob(span_token_ids, token_probs):
    arr = np.array([token_probs.get(int(tid), 1e-8) for tid in span_token_ids], dtype=np.float64)
    log_prob = np.log(arr).sum()
    return float(np.exp(log_prob))

# =========================
# Semantic similarity via SBERT
# =========================
def sbert_sim_batch(sbert_model, anchor_text, doc_texts, normalize=True):
    # Encode anchor once; encode docs in a batch; compute cosine similarities
    a_emb = sbert_model.encode([anchor_text], convert_to_tensor=True, normalize_embeddings=normalize)
    d_emb = sbert_model.encode(doc_texts, convert_to_tensor=True, normalize_embeddings=normalize)
    sims = util.cos_sim(a_emb, d_emb).squeeze(0)  # shape (N,)
    return sims.cpu().tolist()

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def _contains_span(text, span):
    return _norm(span) in _norm(text)

def extract_entities_keywords(text, spacy_model='en_core_web_sm', use_keywords_also=False, num_keywords=5):
    # Load spaCy model
    entities = []
    nlp = spacy.load(spacy_model)
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    if use_keywords_also:
        from keybert import KeyBERT
        # Extract keywords using KeyBERT
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=num_keywords)
        keywords = [kw[0] for kw in keywords]
        entities.extend([(kw, 'KEYPHRASE') for kw in keywords if not _contains_span(text, kw)])

    return {
        'entities': entities
        # 'keywords': keywords
    }
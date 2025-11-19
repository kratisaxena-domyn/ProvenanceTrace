import re
from fuzzywuzzy import fuzz
from icecream import ic

import re

from domynclaimalign.utils.other_utils import jaccard, seqmatch, sbert_sim_batch

class Normalize_Index_Retrieve:

    def normalize_text(self, text):
        return re.sub(r'\s+', ' ', text.lower().strip())

    def normalize_trail(self, trail):
        norm_trail = []
        for entry in trail:
            if isinstance(entry, dict):
                norm_entry = {k: self.normalize_text(str(v)) for k, v in entry.items()}
                norm_trail.append(norm_entry)
            else:
                norm_trail.append(self.normalize_text(str(entry)))
        return norm_trail

    def normalize_entity_keyword_list(self, entity_list):
        return [self.normalize_text(str(e)) for e in entity_list]

    def build_inverted_index(self, trail, entity_list):
        index = {}
        for i, entry in enumerate(trail):
            entry_text = ''
            if isinstance(entry, dict):
                entry_text = ' '.join([str(v) for v in entry.values()])
            else:
                entry_text = str(entry)
            entry_text = self.normalize_text(entry_text)
            for entity in entity_list:
                entity_norm = self.normalize_text(entity)
                if entity_norm in entry_text:
                    index.setdefault(entity_norm, []).append(i)
        return index

    def retrieve_candidate_trail_entries(self, entity_list, inverted_index):
        candidate_indices = set()
        for entity in entity_list:
            entity_norm = self.normalize_text(entity)
            indices = inverted_index.get(entity_norm, [])
            candidate_indices.update(indices)
        return sorted(candidate_indices)

    def fuzzy_retrieve_candidate_trail_entries(self, entity_list, trail, threshold=80):
        candidate_indices = set()
        norm_trail = self.normalize_trail(trail)
        norm_entities = self.normalize_entity_keyword_list(entity_list)
        for i, entry in enumerate(norm_trail):
            entry_text = ''
            if isinstance(entry, dict):
                entry_text = ' '.join([str(v) for v in entry.values()])
            else:
                entry_text = str(entry)
            for entity in norm_entities:
                score = fuzz.partial_ratio(entity, entry_text)
                if score >= threshold:
                    candidate_indices.add(i)
        return sorted(candidate_indices)
    
    def calculate_alignment_score(self, entities, answer, matched_docs, sbert_model=None):
        scores = {}
        # Prepare SBERT similarities if model is provided
        sbert_sims = [0.0] * len(matched_docs)
        if sbert_model is not None and matched_docs:
            doc_texts = []
            for doc in matched_docs:
                if isinstance(doc, dict):
                    doc_texts.append(' '.join([str(v) for v in doc.values()]))
                else:
                    doc_texts.append(str(doc))
            sbert_sims = sbert_sim_batch(sbert_model, answer, doc_texts)
        # Calculate scores for each doc
        for idx, doc in enumerate(matched_docs):
            # Extract entities from doc
            doc_text = ''
            if isinstance(doc, dict):
                doc_text = ' '.join([str(v) for v in doc.values()])
            else:
                doc_text = str(doc)
            # Jaccard score
            doc_entities = set(re.findall(r'\w+', doc_text.lower()))
            query_entities = set([e.lower() for e in entities])
            j = jaccard(doc_entities, query_entities)
            # Description similarity
            desc_sim = seqmatch(' '.join(entities), doc_text)
            # SBERT similarity
            sbert_sim = float(sbert_sims[idx]) if sbert_model is not None else 0.0
            score = 0.5 * j + 0.25 * desc_sim + 0.25 * sbert_sim
            scores[idx] = score
        return scores


    def return_matching_docs(self, entities, trail, answer, sbert_model=None):
        norm_trail = self.normalize_trail(trail)
        norm_entities = self.normalize_entity_keyword_list(entities)
        index = self.build_inverted_index(norm_trail, norm_entities)
        candidates = self.retrieve_candidate_trail_entries(norm_entities, index)
        fuzzy_candidates = self.fuzzy_retrieve_candidate_trail_entries(entities, trail)
        candidates_indices = list(set(candidates + fuzzy_candidates))
        matched_docs = [trail[i] for i in candidates_indices]
        # Filter out answer from matched_docs
        matched_docs_filtered = []
        filtered_indices = []
        for i, doc in zip(candidates_indices, matched_docs):
            if isinstance(doc, dict) and isinstance(answer, dict):
                if doc != answer:
                    matched_docs_filtered.append(doc)
                    filtered_indices.append(i)
            else:
                if str(doc).strip() != str(answer).strip():
                    matched_docs_filtered.append(doc)
                    filtered_indices.append(i)
        scores = self.calculate_alignment_score(entities, answer, matched_docs_filtered, sbert_model)
        # sort matched_docs by descending score
        matched_docs_with_scores = sorted(
            [(doc, scores.get(i, 0)) for i, doc in zip(filtered_indices, matched_docs_filtered)],
            key=lambda x: x[1], reverse=True)
        return matched_docs_with_scores
        
        
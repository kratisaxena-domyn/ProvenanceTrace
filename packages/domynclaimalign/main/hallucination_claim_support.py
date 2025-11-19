

from domynclaimalign.utils.model_utils import load_sentence_transformer #, load_fact_extractor
from domynclaimalign.utils.other_utils import _norm, jaccard, seqmatch
from domynclaimalign.utils.term_list_and_mapping import financial_terms, financial_mapping
from nltk.tokenize import sent_tokenize
from sentence_transformers import util
import spacy
from icecream import ic
import time
import re
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Dict, List, Set
import concurrent.futures
from openai import OpenAI
from functools import partial
import numpy as np
import multiprocessing


# Import config loading - we need to handle the path properly since this is in the package
import json
import os
from pathlib import Path

def load_config_from_examples():
    """Load configuration from the examples/OWFA directory"""
    # Try to find config.json in examples/OWFA directory
    current_dir = Path(__file__).parent
    config_paths = [
        # From package to examples/OWFA
        current_dir.parent.parent.parent / "examples" / "OWFA" / "config.json",
        # From package to examples/OWFA
        current_dir.parent.parent.parent / "examples" / "FAV" / "config.json",
        # Alternative path if run from examples
        Path.cwd() / "config.json",
        # Alternative path if run from examples/OWFA
        Path.cwd() / ".." / "config.json"
    ]
    
    for config_path in config_paths:
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
    
    # Fallback to default config
    print("Warning: config.json not found, using default configuration")
    return {
        "llm_client": {
            "base_url": "<Your_Model_API_Endpoint_URL>",
            "api_key": "EMPTY",
            "model_name": "iGenius-AI-Team/Domyn-Small"
        }
    }

def get_llm_client_config():
    """Get LLM client configuration"""
    config = load_config_from_examples()
    return config.get("llm_client", {
        "base_url": "<Your_Model_API_Endpoint_URL>",
        "api_key": "EMPTY",
        "model_name": "iGenius-AI-Team/Domyn-Small"
    })

# Initialize OpenAI client with config
llm_config = get_llm_client_config()
client = OpenAI(
    base_url=llm_config["base_url"],
    api_key=llm_config["api_key"]
)
MODEL_NAME = llm_config["model_name"]


# class TraceNormalizer:
# 	def __init__(self, spacy_model='en_core_web_sm'):
# 		self.nlp = spacy.load(spacy_model)

# 	def dict_to_text(self, entry):
# 		if isinstance(entry, dict):
# 			return ' '.join([f"{k}:{v}" for k, v in entry.items()])
# 		return str(entry)

# 	def normalize_list(self, input_list):
# 		return [self.dict_to_text(entry) for entry in input_list]

# 	def dedup_segment(self, text_list):
# 		merged_text = ' '.join(text_list)
# 		doc = self.nlp(merged_text)
# 		sents = list(set([sent.text.strip() for sent in doc.sents if sent.text.strip()]))
# 		return sents


class TraceNormalizer:
    def __init__(self, spacy_model='en_core_web_sm'):
        self.nlp = spacy.load(spacy_model)
        self.max_chunk_size = 900000  # Keep under 1M limit
        self.max_workers = min(16, multiprocessing.cpu_count())  # Limit workers for spaCy
    
    def _process_chunk(self, chunk):
        """Process a single chunk of text with length validation."""
        try:
            # Double-check chunk length before processing
            if len(chunk) > self.nlp.max_length:
                # If still too long, use regex fallback
                sentences = re.split(r'[.!?]+', chunk)
                return [sent.strip() for sent in sentences if sent.strip()]
            
            doc = self.nlp(chunk)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        except Exception as e:
            # Fallback to simple sentence splitting if spaCy fails
            import re
            sentences = re.split(r'[.!?]+', chunk)
            return [sent.strip() for sent in sentences if sent.strip()]
        
    def process_large_text(self, text):
        """Process large text by chunking it with parallel processing."""
        if len(text) <= self.nlp.max_length:
            return self.nlp(text)
        
        # Split into chunks
        chunks = []
        for i in range(0, len(text), self.max_chunk_size):
            chunk = text[i:i + self.max_chunk_size]
            chunks.append(chunk)
        
        # Process chunks in parallel
        if len(chunks) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                processed_chunks = list(executor.map(lambda chunk: self.nlp(chunk), chunks))
            return processed_chunks
        else:
            return [self.nlp(chunks[0])]
    
    def dedup_segment(self, text_list):
        """Optimized deduplication with parallel processing and length checking."""
        if not text_list:
            return []
        
        # Calculate total length without creating the full string
        total_length = sum(len(text) + 1 for text in text_list)  # +1 for space
        
        if total_length <= self.nlp.max_length:
            # Safe to process normally
            merged_text = ' '.join(text_list)
            doc = self.nlp(merged_text)
            all_sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            # Process in chunks without creating full merged text
            all_sentences = []
            current_chunk = []
            current_length = 0
            
            for text in text_list:
                text_len = len(text) + 1  # +1 for space
                
                if current_length + text_len > self.max_chunk_size and current_chunk:
                    # Process current chunk
                    chunk_text = ' '.join(current_chunk)
                    chunk_sentences = self._process_chunk(chunk_text)
                    all_sentences.extend(chunk_sentences)
                    
                    # Start new chunk
                    current_chunk = [text]
                    current_length = text_len
                else:
                    current_chunk.append(text)
                    current_length += text_len
            
            # Process final chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk_sentences = self._process_chunk(chunk_text)
                all_sentences.extend(chunk_sentences)
        
        # Use set for O(1) deduplication
        return list(set(all_sentences))
    
    def dedup_segment_batch(self, text_lists):
        """Process multiple text lists in parallel."""
        if not text_lists:
            return []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self.dedup_segment, text_lists))
        
        return results
    
    def dict_to_text(self, entry):
        if isinstance(entry, dict):
            return ' '.join([f"{k}:{v}" for k, v in entry.items()])
        return str(entry)

    def normalize_list(self, input_list):
        return [self.dict_to_text(entry) for entry in input_list]

class FinancialTermNormalizer:
    def __init__(self):
        self.static_mappings = self._load_static_mappings()
        self.learned_mappings = {}
        
    def _load_static_mappings(self) -> Dict[str, List[str]]:
        """Load core financial term mappings that are commonly used."""
        return financial_mapping
        # return {
        #     # Core ratios
        #     'pe': ['p/e', 'price earnings', 'price_earnings', 'pe_ratio'],
        #     'pb': ['p/b', 'price book', 'price_book', 'pb_ratio'],
        #     'roe': ['return on equity', 'return_on_equity'],
        #     'roa': ['return on assets', 'return_on_assets'],
        #     'eps': ['earnings per share', 'earnings_per_share'],
            
        #     # Basic metrics
        #     'revenue': ['sales', 'turnover', 'income'],
        #     'ebitda': ['earnings before interest tax depreciation amortization'],
        #     'volume': ['vol', 'trading_volume'],
        #     'market_cap': ['market capitalization', 'marketcap'],
        # }
    
    @lru_cache(maxsize=1000)
    def _extract_financial_keywords(self, text: str) -> Set[str]:
        """Extract potential financial keywords from text using patterns."""
        text = text.lower()
        
        # Patterns for financial metrics
        patterns = [
            r'(\w+)\s*ratio',           # Something ratio
            r'(\w+)\s*margin',          # Something margin  
            r'(\w+)\s*rate',            # Something rate
            r'(\w+)\s*yield',           # Something yield
            r'(\w+)\s*multiple',        # Something multiple
            r'(\w+)\s*coverage',        # Something coverage
            r'(\w+)\s*turnover',        # Something turnover
            r'return\s+on\s+(\w+)',     # Return on something
            r'(\w+)\s+per\s+share',     # Something per share
            r'(\w+)\s+to\s+(\w+)',      # Something to something
            r'(\w+)\s+growth',          # Something growth
            r'(\w+)\s+value',           # Something value
            r'(\w+)\s+price',           # Something price
            r'(\w+)\s+cost',            # Something cost
        ]
        
        keywords = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        keywords.update(match)
                    else:
                        keywords.add(match)
        
        # Also extract standalone financial terms
        financial_indicators = [
            'pe', 'pb', 'ps', 'ev', 'ebitda', 'ebit', 'fcf', 'roe', 'roa', 'roic',
            'debt', 'equity', 'assets', 'liabilities', 'revenue', 'profit', 'loss',
            'dividend', 'beta', 'alpha', 'volatility', 'correlation', 'sharpe'
        ]
        
        words = re.findall(r'\w+', text)
        for word in words:
            if word in financial_indicators:
                keywords.add(word)
        
        return keywords
    
    def _create_variations(self, term: str) -> List[str]:
        """Create common variations of a financial term."""
        variations = [term]
        
        # CamelCase variations
        if '_' in term:
            camel_case = ''.join(word.capitalize() for word in term.split('_'))
            variations.append(camel_case.lower())
            variations.append(camel_case[0].lower() + camel_case[1:])
        
        # Underscore variations
        if ' ' in term:
            variations.append(term.replace(' ', '_'))
        
        # Slash variations for ratios
        if 'to' in term:
            variations.append(term.replace(' to ', '/'))
            variations.append(term.replace(' to ', '_'))
        
        # Abbreviation attempts
        words = term.split()
        if len(words) > 1:
            abbreviation = ''.join(word[0] for word in words)
            variations.append(abbreviation)
        
        return variations
    
    @lru_cache(maxsize=500)
    def normalize_term(self, term: str) -> str:
        """Normalize a financial term to its canonical form."""
        term_lower = term.lower().strip()
        
        # Check static mappings first
        for canonical, variants in self.static_mappings.items():
            if term_lower == canonical or term_lower in variants:
                return canonical
        
        # Check learned mappings
        for canonical, variants in self.learned_mappings.items():
            if term_lower == canonical or term_lower in variants:
                return canonical
        
        # Extract and normalize financial keywords
        keywords = self._extract_financial_keywords(term_lower)
        if keywords:
            # Use the longest keyword as the canonical form
            canonical = max(keywords, key=len) if keywords else term_lower
            
            # Learn this mapping
            if canonical not in self.learned_mappings:
                self.learned_mappings[canonical] = self._create_variations(term_lower)
            
            return canonical
        
        return term_lower


class HallucinationClaimSupport_advanced:
    def __init__(self, spacy_model='en_core_web_sm', k=5, threshold=0.4, context_window=15):
        self.nlp = spacy.load(spacy_model)
        self.k = k
        self.threshold = threshold
        self.normalizer = TraceNormalizer(spacy_model)
        self.context_window = context_window
        self.term_normalizer = FinancialTermNormalizer()

    def _normalize_financial_terms(self, text):
        """Enhanced financial term normalization using the dynamic normalizer."""
        # Extract potential financial terms
        doc = self.nlp(text)
        
        # Look for noun phrases that might be financial metrics
        financial_phrases = []
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower()
            if any(keyword in chunk_text for keyword in ['ratio', 'rate', 'margin', 'return', 'value', 'price', 'cost', 'yield']):
                financial_phrases.append(chunk_text)
        
        # Also look for patterns with numbers
        financial_patterns = re.findall(r'(\w+(?:\s+\w+)*)\s*(?:is|of|at|equals?)\s*\d+', text.lower())
        financial_phrases.extend(financial_patterns)
        
        # Normalize the text
        normalized_text = text.lower()
        for phrase in financial_phrases:
            canonical = self.term_normalizer.normalize_term(phrase)
            normalized_text = normalized_text.replace(phrase, canonical)
        
        return normalized_text

    def _extract_numeric_values_enhanced(self, text):
        """Extract numbers with enhanced pattern matching and conversion."""
        patterns = [
            r'\d+(?:\.\d+)?%',  # Percentages
            r'\d+(?:\.\d+)?',   # Regular numbers
            r'\$\d+(?:\.\d+)?[KMB]?',  # Currency with K/M/B suffixes
            r'\d+(?:\.\d+)?[KMB]',     # Numbers with K/M/B suffixes
        ]
        
        matches = []
        for pattern in patterns:
            found = re.findall(pattern, text)
            matches.extend(found)
        
        return matches

    def _convert_to_comparable_values(self, value_str):
        """Convert various number formats to comparable float values."""
        try:
            value_str = value_str.strip().lower()
            
            # Handle percentages - convert to decimal
            if '%' in value_str:
                num = float(value_str.replace('%', ''))
                return num / 100.0, 'percentage'
            
            # Handle currency symbols
            if '$' in value_str:
                value_str = value_str.replace('$', '')
            
            # Handle suffixes (K, M, B)
            multipliers = {'k': 1000, 'm': 1000000, 'b': 1000000000}
            for suffix, mult in multipliers.items():
                if value_str.endswith(suffix):
                    num = float(value_str[:-1])
                    return num * mult, 'scaled'
            
            # Regular number
            return float(value_str), 'decimal'
            
        except (ValueError, AttributeError):
            return None, None

    def _are_values_equivalent(self, claim_val, support_val, tolerance=0.0): # TODO: tune the tolerance, earlier 0.15
        """Check if two values are equivalent considering various conversions."""
        claim_num, claim_type = self._convert_to_comparable_values(claim_val)
        support_num, support_type = self._convert_to_comparable_values(support_val)
        
        if claim_num is None or support_num is None:
            return False
        
        # Direct comparison
        if abs(claim_num - support_num) / max(abs(claim_num), abs(support_num), 1e-10) <= tolerance:
            return True
        
        # Check percentage vs decimal conversion
        if claim_type == 'percentage' and support_type == 'decimal':
            # Claim is percentage, support might be decimal (e.g., 0.1534 vs 15.34%)
            if abs(claim_num - support_num) / max(abs(claim_num), abs(support_num), 1e-10) <= tolerance:
                return True
        elif claim_type == 'decimal' and support_type == 'percentage':
            # Support is percentage, claim might be decimal
            if abs(claim_num - support_num) / max(abs(claim_num), abs(support_num), 1e-10) <= tolerance:
                return True
        
        # Check if one needs to be converted to percentage
        if claim_type == 'decimal' and support_type == 'decimal':
            # Try converting claim to percentage (multiply by 100)
            claim_as_percent = claim_num * 100
            if abs(claim_as_percent - support_num) / max(abs(claim_as_percent), abs(support_num), 1e-10) <= tolerance:
                return True
            
            # Try converting support to percentage
            support_as_percent = support_num * 100
            if abs(claim_num - support_as_percent) / max(abs(claim_num), abs(support_as_percent), 1e-10) <= tolerance:
                return True
        
        return False

    def _extract_contextual_information(self, text, target_numbers, context_tokens=15):
        """Extract contextual spans around target numbers with enhanced context."""
        matches = []
        tokens = text.split()
        
        # First, find all numeric values in the text
        for i, token in enumerate(tokens):
            token_numbers = self._extract_numeric_values_enhanced(token)
            
            for token_num in token_numbers:
                # Check if this number is relevant to our target numbers
                is_relevant = any(self._are_values_equivalent(token_num, target_num) 
                                for target_num in target_numbers)
                
                if is_relevant:
                    # Extract broader context
                    start_idx = max(0, i - context_tokens)
                    end_idx = min(len(tokens), i + context_tokens + 1)
                    context_span = " ".join(tokens[start_idx:end_idx])
                    
                    # Extract key-value context if it looks like structured data
                    key_context = self._extract_key_value_context(tokens, i)
                    
                    matches.append({
                        'number': token_num,
                        'context': context_span,
                        'key_context': key_context,
                        'position': i,
                        'relevance_score': self._calculate_relevance_score(token_num, target_numbers)
                    })
        
        return matches

    def _extract_key_value_context(self, tokens, number_position):
        """Extract key-value pair context around a number position."""
        # Look for patterns like 'key': value or key: value around the number
        start_search = max(0, number_position - 5)
        end_search = min(len(tokens), number_position + 5)
        
        context_tokens = tokens[start_search:end_search]
        context_text = " ".join(context_tokens)
        
        # Try to extract key-value patterns
        kv_patterns = [
            r"'([^']+)':\s*[^\s,}]+",
            r'"([^"]+)":\s*[^\s,}]+', 
            r'(\w+):\s*[^\s,}]+'
        ]
        
        for pattern in kv_patterns:
            matches = re.findall(pattern, context_text)
            if matches:
                return matches[0] if isinstance(matches[0], str) else matches[0][0]
        
        return None

    def _calculate_relevance_score(self, found_number, target_numbers):
        """Calculate how relevant a found number is to target numbers."""
        max_relevance = 0.0
        
        for target in target_numbers:
            if self._are_values_equivalent(found_number, target, tolerance=0.2):
                # Exact or very close match
                relevance = 1.0
            else:
                # Calculate similarity based on magnitude
                found_val, _ = self._convert_to_comparable_values(found_number)
                target_val, _ = self._convert_to_comparable_values(target)
                
                if found_val is not None and target_val is not None:
                    if max(found_val, target_val) > 0:
                        diff_ratio = abs(found_val - target_val) / max(found_val, target_val)
                        relevance = max(0, 1.0 - diff_ratio)
                    else:
                        relevance = 1.0 if found_val == target_val else 0.0
                else:
                    relevance = 0.0
            
            max_relevance = max(max_relevance, relevance)
        
        return max_relevance

    def _calculate_semantic_similarity(self, claim, context, sbert_model=None):
        """Calculate semantic similarity between claim and context using sentence transformers."""
        if sbert_model is None:
            # Fallback to string similarity if no model provided
            return SequenceMatcher(None, claim.lower(), context.lower()).ratio()
        
        try:
            # Use sentence transformer for semantic similarity
            claim_emb = sbert_model.encode([claim], convert_to_tensor=True, normalize_embeddings=True)
            context_emb = sbert_model.encode([context], convert_to_tensor=True, normalize_embeddings=True)
            similarity = util.cos_sim(claim_emb, context_emb).item()
            return similarity
        except:
            # Fallback to string similarity
            return SequenceMatcher(None, claim.lower(), context.lower()).ratio()

    def _calculate_comprehensive_similarity(self, claim, context_info, sbert_model=None):
        """Enhanced similarity calculation with semantic understanding."""
        claim_norm = self._normalize_financial_terms(claim)
        context_norm = self._normalize_financial_terms(context_info['context'])
        
        # 1. Semantic similarity using sentence transformers
        semantic_sim = self._calculate_semantic_similarity(claim, context_info['context'], sbert_model)
        
        # 2. Normalized text similarity
        text_sim = SequenceMatcher(None, claim_norm, context_norm).ratio()
        
        # 3. Word overlap after normalization
        claim_words = set(claim_norm.split())
        context_words = set(context_norm.split())
        word_overlap = jaccard(claim_words, context_words)
        
        # 4. Financial term matching
        claim_terms = self.term_normalizer._extract_financial_keywords(claim_norm)
        context_terms = self.term_normalizer._extract_financial_keywords(context_norm)
        term_overlap = jaccard(claim_terms, context_terms) if claim_terms or context_terms else 0.0
        
        # 5. Key concept matching
        key_context = context_info.get('key_context', '')
        if key_context:
            key_norm = self.term_normalizer.normalize_term(key_context)
            key_sim = SequenceMatcher(None, claim_norm, key_norm).ratio()
        else:
            key_sim = 0.0
        
        # 6. Number relevance score
        number_relevance = context_info.get('relevance_score', 0.0)
        
        # Weighted combination with emphasis on semantic similarity
        # comprehensive_score = (
        #     0.35 * semantic_sim +      # Increased weight for semantic similarity
        #     0.15 * text_sim +
        #     0.15 * word_overlap +
        #     0.15 * term_overlap +      # Financial term matching
        #     0.1 * key_sim +
        #     0.1 * number_relevance
        # )
        # TODO: tune the weights
        comprehensive_score = (
            0.25 * semantic_sim +      # Increased weight for semantic similarity
            0.1 * text_sim +
            0.1 * word_overlap +
            0.2 * term_overlap +      # Financial term matching
            0.1 * key_sim +
            0.25 * number_relevance
        )
        
        return min(1.0, comprehensive_score)

    def _classify_claim(self, claim):
        """Enhanced claim classification."""
        # Check for numeric content
        numbers = self._extract_numeric_values_enhanced(claim)
        
        # Check for financial/technical terms
        # financial_terms = ['ratio', 'pe', 'margin', 'rate', 'cap', 'volume', 'revenue', 
        #                   'profit', 'earnings', 'price', 'value', 'percent', '%']
        
        has_financial_terms = any(term in claim.lower() for term in financial_terms)
        
        if numbers and has_financial_terms:
            return 'numeric'
        # elif numbers:
        #     return 'numeric'
        else:
            return 'generic'
        
    def _find_enhanced_matches(self, claim, support_sents, sbert_model=None):
        """Optimized enhanced matching with parallelization and reduced complexity."""
        
        
        # Early return for empty inputs
        if not support_sents:
            return []
        
        start_time = time.time()
        # Pre-extract claim numbers once
        claim_numbers = self._extract_numeric_values_enhanced(claim)
        is_numeric_claim = bool(claim_numbers)
        preextract_time = time.time() - start_time
        ic(preextract_time)
        
        # Pre-process all support sentences in parallel
        def process_sentence(args):
            sent_idx, sent = args
            context_matches = []
            
            # Extract contextual information for numeric claims only
            if is_numeric_claim:
                context_matches = self._extract_contextual_information(sent, claim_numbers)
            
            # Process JSON data - extract once per sentence
            json_processed = self._extract_financial_metrics_from_json(sent)
            
            # Process JSON matches based on claim type
            if is_numeric_claim:
                # Batch process all JSON numbers for efficiency
                for json_data in json_processed:
                    json_numbers = self._extract_numeric_values_enhanced(json_data['value'])
                    if json_numbers:
                        # Use set intersection for faster relevance checking
                        relevant_numbers = [
                            json_num for json_num in json_numbers
                            if any(self._are_values_equivalent(json_num, claim_num) 
                                for claim_num in claim_numbers)
                        ]
                        
                        for json_num in relevant_numbers:
                            context_matches.append({
                                'number': json_num,
                                'context': json_data['sentence'],
                                'key_context': json_data['readable_key'],
                                'position': 0,
                                'relevance_score': self._calculate_relevance_score(json_num, claim_numbers)
                            })
            else:
                # For non-numeric claims, add all JSON data with pre-calculated scores
                for json_data in json_processed:
                    context_matches.append({
                        'number': None,
                        'context': json_data['sentence'],
                        'key_context': json_data['readable_key'],
                        'position': 0,
                        'relevance_score': 0.5
                    })
            
            return sent_idx, sent, context_matches
        
        start_time = time.time()
        # Parallel processing of sentences
        max_workers = min(8, len(support_sents))  # Limit workers to avoid overhead
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            sentence_args = list(enumerate(support_sents))
            processed_sentences = list(executor.map(process_sentence, sentence_args))

        processing_time = time.time() - start_time
        ic(processing_time)
        
        start_time = time.time()
        # Batch collect all contexts for similarity calculation
        all_contexts = []
        context_metadata = []
        
        for sent_idx, sent, context_matches in processed_sentences:
            for match in context_matches:
                all_contexts.append(match['context'])
                context_metadata.append({
                    'sent_idx': sent_idx,
                    'support_sent': sent,
                    'match_data': match
                })
        context_metadata_time = time.time() - start_time
        ic(context_metadata_time)
        
        # Early return if no contexts found
        if not all_contexts:
            return []
        
        start_time = time.time()
        # Batch calculate similarities using sentence transformer if available
        if sbert_model and len(all_contexts) > 1:
            try:
                # Batch encode all contexts at once for efficiency
                claim_emb = sbert_model.encode([claim], convert_to_tensor=True, normalize_embeddings=True)
                context_embs = sbert_model.encode(all_contexts, convert_to_tensor=True, normalize_embeddings=True)
                similarities = util.cos_sim(claim_emb, context_embs).cpu().numpy()[0]
                
                # Build results with batch-computed similarities
                all_matches = []
                for i, (similarity_score, metadata) in enumerate(zip(similarities, context_metadata)):
                    match_data = metadata['match_data']
                    all_matches.append({
                        'sent_idx': metadata['sent_idx'],
                        'support_sent': metadata['support_sent'],
                        'context': match_data['context'],
                        'similarity_score': float(similarity_score),
                        'number_relevance': match_data['relevance_score'],
                        'key_context': match_data.get('key_context', ''),
                        'matched_number': match_data.get('number', '')
                    })
            except Exception:
                # Fallback to individual similarity calculations
                all_matches = []
                for metadata in context_metadata:
                    match_data = metadata['match_data']
                    similarity_score = self._calculate_comprehensive_similarity(claim, match_data, sbert_model)
                    all_matches.append({
                        'sent_idx': metadata['sent_idx'],
                        'support_sent': metadata['support_sent'],
                        'context': match_data['context'],
                        'similarity_score': similarity_score,
                        'number_relevance': match_data['relevance_score'],
                        'key_context': match_data.get('key_context', ''),
                        'matched_number': match_data.get('number', '')
                    })
        else:
            # Fallback when no sentence transformer model is available
            all_matches = []
            for metadata in context_metadata:
                match_data = metadata['match_data']
                similarity_score = self._calculate_comprehensive_similarity(claim, match_data, sbert_model)
                all_matches.append({
                    'sent_idx': metadata['sent_idx'],
                    'support_sent': metadata['support_sent'],
                    'context': match_data['context'],
                    'similarity_score': similarity_score,
                    'number_relevance': match_data['relevance_score'],
                    'key_context': match_data.get('key_context', ''),
                    'matched_number': match_data.get('number', '')
                })
        similarities_time = time.time() - start_time
        ic(similarities_time)
        
        # Sort by similarity score (use numpy argsort for better performance on large datasets)
        if len(all_matches) > 100:
            scores = np.array([match['similarity_score'] for match in all_matches])
            sorted_indices = np.argsort(scores)[::-1]
            all_matches = [all_matches[i] for i in sorted_indices]
        else:
            all_matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return all_matches

    def _extract_financial_metrics_from_json(self, text):
        """Extract financial metrics from JSON-like data with enhanced pattern matching."""
        processed_data = []
        
        # Enhanced patterns for various formats
        patterns = [
            r"'([^']+)':\s*([^,}]+)",  # 'key': value
            r'"([^"]+)":\s*([^,}]+)',  # "key": value
            r'(\w+):\s*([^,}\s]+)',    # key: value
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for key, value in matches:
                # Clean up value
                value = value.strip().rstrip(',').strip('"\'')
                
                # Normalize the key using our financial term normalizer
                normalized_key = self.term_normalizer.normalize_term(key)
                readable_key = re.sub(r'([a-z])([A-Z])', r'\1 \2', normalized_key).title()
                
                # Create multiple sentence variations
                sentences = [
                    f"The {readable_key} is {value}.",
                    f"{readable_key}: {value}",
                    f"{normalized_key} equals {value}",
                    f"{key} is {value}"  # Keep original key as well
                ]
                
                for sentence in sentences:
                    processed_data.append({
                        'original_key': key,
                        'normalized_key': normalized_key,
                        'value': value,
                        'readable_key': readable_key,
                        'sentence': sentence
                    })
        
        return processed_data

    def _preprocess_json_data(self, text):
        """Use the enhanced financial metrics extraction."""
        return self._extract_financial_metrics_from_json(text)

    def _process_numeric_claims(self, numeric_claims, support_sents, sbert_model):
        """Optimized numeric claim processing with batch operations."""
        if not numeric_claims:
            return []
        
        results = []
        
        # Pre-encode all support sentences once for fallback similarity
        doc_embs = sbert_model.encode(support_sents, convert_to_tensor=True, show_progress_bar=False)
        
        start_time = time.time()
        # Batch process claims to reduce overhead
        for claim in numeric_claims:
            # Find enhanced matches with semantic understanding
            enhanced_matches = self._find_enhanced_matches(claim, support_sents, sbert_model)
            
            if enhanced_matches:
                # Use enhanced matches - limit to top 10 for efficiency
                top_matches = enhanced_matches[:10]
                
                # Pre-calculate similarity scores to avoid repeated access
                similarity_scores = [match['similarity_score'] for match in top_matches]
                
                topk_sents = [(match['context'], score) for match, score in zip(top_matches, similarity_scores)]
                max_sim = max(similarity_scores)
                avg_sim = sum(similarity_scores) / len(similarity_scores)
                
                # Adjusted threshold for semantic similarity
                status = "Supported" if max_sim >= self.threshold else "Potential Hallucination"
                ## TODO: was 0.4 a better threshold earlier? Also see we are using max_sim for numerical claims
                result_entry = {
                    "claim": claim,
                    "claim_type": "numeric",
                    "status": status,
                    "max_sim": max_sim,
                    "avg_topk_sim": avg_sim,
                    "topk_sents": topk_sents,
                    "enhanced_matches": len(enhanced_matches),
                    "matching_details": top_matches[:3]
                }
            else:
                # Fallback to semantic similarity using pre-computed embeddings
                claim_emb = sbert_model.encode([claim], convert_to_tensor=True, show_progress_bar=False)
                sims = util.cos_sim(claim_emb, doc_embs).cpu().numpy()[0]
                
                # Use numpy operations for efficiency
                topk_idx = sims.argsort()[-self.k:][::-1]
                topk_sims = sims[topk_idx]
                
                topk_sents = [(support_sents[j], float(topk_sims[i])) for i, j in enumerate(topk_idx)]
                max_sim = float(topk_sims[0]) if len(topk_idx) > 0 else 0.0
                avg_sim = float(topk_sims.mean()) if len(topk_idx) > 0 else 0.0

                status = "Supported" if avg_sim >= self.threshold else "Potential Hallucination"
                ## TODO: since there are no enhanced matches, we use avg_sim for numerical claims
                result_entry = {
                    "claim": claim,
                    "claim_type": "numeric",
                    "status": status,
                    "max_sim": max_sim,
                    "avg_topk_sim": avg_sim,
                    "topk_sents": topk_sents,
                    "enhanced_matches": 0,
                    "matching_details": []
                }
            
            results.append(result_entry)
        extraction_time = time.time() - start_time
        ic(extraction_time)
        return results

    def _process_generic_claims(self, generic_claims, support_sents, sbert_model):
        """Process generic claims with enhanced preprocessing."""
        if not generic_claims:
            return []
        
        # Enhanced preprocessing
        processed_support_sents = []
        for sent in support_sents:
            # Add original sentence
            processed_support_sents.append(sent)
            
            # Add processed JSON variations
            json_data = self._extract_financial_metrics_from_json(sent)
            for data in json_data:
                processed_support_sents.append(data['sentence'])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_processed = []
        for sent in processed_support_sents:
            if sent not in seen:
                seen.add(sent)
                unique_processed.append(sent)
        
        # Encode and calculate similarities
        claim_embs = sbert_model.encode(generic_claims, convert_to_tensor=True, show_progress_bar=False)
        doc_embs = sbert_model.encode(unique_processed, convert_to_tensor=True, show_progress_bar=False)
        sim_matrix = util.cos_sim(claim_embs, doc_embs).cpu().numpy()
        
        results = []
        
        for i, claim in enumerate(generic_claims):
            sims = sim_matrix[i]
            topk_idx = sims.argsort()[-self.k:][::-1]
            topk_sents = [(unique_processed[j], float(sims[j])) for j in topk_idx]
            
            max_sim = float(sims[topk_idx[0]]) if len(topk_idx) > 0 else 0.0
            avg_sim = float(sims[topk_idx].mean()) if len(topk_idx) > 0 else 0.0
            
            status = "Supported" if avg_sim >= self.threshold else "Potential Hallucination"
            
            result_entry = {
                "claim": claim,
                "claim_type": "generic",
                "status": status,
                "max_sim": max_sim,
                "avg_topk_sim": avg_sim,
                "topk_sents": topk_sents
            }
            
            results.append(result_entry)
        
        return results

    def check_claims(self, claims, support_sents, sbert_model):
        # Normalize and deduplicate support sentences
        support_sents = self.normalizer.normalize_list(support_sents)
        support_sents = self.normalizer.dedup_segment(support_sents)
        
        # Deduplicate claims
        claims = list(set([c.strip() for c in claims if c.strip()]))
        
        if not claims or not support_sents:
            return []
        
        # Enhanced claim classification
        numeric_claims = []
        generic_claims = []
        
        for claim in claims:
            claim_type = self._classify_claim(claim)
            if claim_type == 'numeric':
                numeric_claims.append(claim)
            else:
                generic_claims.append(claim)
        
        # Process each bucket
        results = []
        
        if numeric_claims:
            numeric_results = self._process_numeric_claims(numeric_claims, support_sents, sbert_model)
            results.extend(numeric_results)
        
        if generic_claims:
            generic_results = self._process_generic_claims(generic_claims, support_sents, sbert_model)
            results.extend(generic_results)
        
        # Step 3: Run entailment on all results
        if results:
            results = check_entailment_domyn_small_with_mapping(results)
        
        return results



# for flask app with sentence mapping

def check_entailment_domyn_small_with_mapping(claims_with_topk, use_only_entailment=False, max_workers=16):
    """
    claims_with_topk: List of dicts, each with keys:
        - 'claim': str
        - 'sent_idx': int
        - 'sent': str
        - 'topk_sents': list of (sent, sim_score)
    Returns: List of dicts, each with:
        - 'claim'
        - 'sent_idx': int
        - 'sent': str
        - 'topk_sents': list of (sent, sim_score)
        - 'entailment_score': float (single score for all topk_sents)
    """
    def get_entailment_score(claim, merged_sents):
        prompt = (
            f"Do the following sentences together support the claim?\n"
            f"Claim: {claim}\n"
            f"Sentences: {merged_sents}\n"
            f"Answer with only a number between 0 (no support) and 1 (full support). For example, if the claim is fully supported, answer: 1. If not supported at all, answer: 0. And nothing else"
        )
        result = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an atomic fact extractor."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=16,
            temperature=0.2,
            extra_body={
                "top_k": 1,
                "chat_template_kwargs": {"enable_thinking": False}
            }
        )
        try:
            score = float(result.choices[0].message.content.strip())
            return score
        except Exception:
            return 1.0

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        # merged_inputs = []
        for entry in claims_with_topk:
            claim = entry['claim']
            # Merge all topk_sents into a single text
            merged_sents = " ".join([sent for sent, _ in entry['topk_sents']])
            # merged_inputs.append((claim, merged_sents))
            futures.append(executor.submit(get_entailment_score, claim, merged_sents))
        entailment_scores = [f.result() for f in futures]

    for idx, entry in enumerate(claims_with_topk):
        topk = [(sent, sim_score) for sent, sim_score in entry['topk_sents']]
        # keep only unique sentencses in topk
        seen_sents = set()
        unique_topk = []
        for sent, sim_score in topk:
            if sent not in seen_sents:
                seen_sents.add(sent)
                unique_topk.append((sent, sim_score))
        topk = unique_topk
        entailment_score = entailment_scores[idx]
        # change status to "Supported" if entailment_score > 0.5
        if entailment_score >= 0.5:
            entry['status'] = "Supported"
        if use_only_entailment and entailment_score < 0.5:
            entry['status'] = "Potential Hallucination"
        results.append({
            'claim': entry['claim'],
            'sent_idx': entry.get('sent_idx'),
            'sent': entry.get('sent'),
            'status': entry.get('status', ''),
            'max_sim': entry.get('max_sim', 0.0),
            'avg_topk_sim': entry.get('avg_topk_sim', 0.0),
            'topk_sents': topk,
            'entailment_score': entailment_score
        })
    return results

# Advanced class with mapping (sent and sent_idx returned)
class HallucinationClaimSupport_advanced_withMapping(HallucinationClaimSupport_advanced):
    def check_claims(self, fact_mappings, support_sents, sbert_model, use_numerical=True, use_only_entailment=False):
        # Normalize and deduplicate support sentences
        support_sents = self.normalizer.normalize_list(support_sents)
        support_sents = self.normalizer.dedup_segment(support_sents)

        start_time = time.time()
        # Extract unique facts while preserving mapping info
        unique_fact_mappings = []
        seen_facts = set()
        for mapping in fact_mappings:
            fact = mapping["fact"].strip()
            if fact and fact not in seen_facts:
                seen_facts.add(fact)
                unique_fact_mappings.append(mapping)
        if not unique_fact_mappings or not support_sents:
            return []
        mapping_time = time.time() - start_time
        ic(mapping_time)
        
        # Enhanced claim classification with mapping
        
        if use_numerical:
            start_time = time.time()
            numeric_claims = []
            generic_claims = []
            numeric_mappings = []
            generic_mappings = []
            for mapping in unique_fact_mappings:
                claim = mapping["fact"]
                claim_type = self._classify_claim(claim)
                if claim_type == 'numeric':
                    numeric_claims.append(claim)
                    numeric_mappings.append(mapping)
                else:
                    generic_claims.append(claim)
                    generic_mappings.append(mapping)
            classification_time = time.time() - start_time
            ic(classification_time)
            
            start_time = time.time()
            results = []

            # --- Optimization: Batch encode all unique sentences for generic claims ---
            if generic_claims:
                # Preprocess all support sentences and JSON variations only once
                processed_support_sents = []
                for sent in support_sents:
                    processed_support_sents.append(sent)
                    json_data = self._extract_financial_metrics_from_json(sent)
                    for data in json_data:
                        processed_support_sents.append(data['sentence'])
                # Remove duplicates while preserving order
                seen = set()
                unique_processed = []
                for sent in processed_support_sents:
                    if sent not in seen:
                        seen.add(sent)
                        unique_processed.append(sent)
                # Batch encode all claims and all unique processed support sentences
                claim_embs = sbert_model.encode(generic_claims, convert_to_tensor=True, show_progress_bar=False)
                doc_embs = sbert_model.encode(unique_processed, convert_to_tensor=True, show_progress_bar=False)
                sim_matrix = util.cos_sim(claim_embs, doc_embs).cpu().numpy()
                generic_results = []
                for i, claim in enumerate(generic_claims):
                    sims = sim_matrix[i]
                    topk_idx = sims.argsort()[-self.k:][::-1]
                    topk_sents = [(unique_processed[j], float(sims[j])) for j in topk_idx]
                    max_sim = float(sims[topk_idx[0]]) if len(topk_idx) > 0 else 0.0
                    avg_sim = float(sims[topk_idx].mean()) if len(topk_idx) > 0 else 0.0
                    status = "Supported" if avg_sim >= self.threshold else "Potential Hallucination"
                    result_entry = {
                        "claim": claim,
                        "claim_type": "generic",
                        "status": status,
                        "max_sim": max_sim,
                        "avg_topk_sim": avg_sim,
                        "topk_sents": topk_sents
                    }
                    generic_results.append(result_entry)
                # Attach sent and sent_idx from mapping
                for res, mapping in zip(generic_results, generic_mappings):
                    res["sent_idx"] = mapping.get("sent_idx")
                    res["sent"] = mapping.get("sent")
                    results.append(res)
            generic_claims_time = time.time() - start_time
            ic(generic_claims_time)
            
            start_time = time.time()
            # --- Optimization: Parallelize numeric claim processing ---
            if numeric_claims:
                from concurrent.futures import ThreadPoolExecutor
                def process_numeric_one(args):
                    claim, mapping = args
                    numeric_result = self._process_numeric_claims([claim], support_sents, sbert_model)[0]
                    numeric_result["sent_idx"] = mapping.get("sent_idx")
                    numeric_result["sent"] = mapping.get("sent")
                    return numeric_result
                with ThreadPoolExecutor() as executor:
                    numeric_results = list(executor.map(process_numeric_one, zip(numeric_claims, numeric_mappings)))
                results.extend(numeric_results)
            numeric_claims_time = time.time() - start_time
            ic(numeric_claims_time)
        else:
            start_time = time.time()
            results = []
            generic_claims = []
            generic_mappings = []
            for mapping in unique_fact_mappings:
                claim = mapping["fact"]
                generic_claims.append(claim)
                generic_mappings.append(mapping)
  
            processed_support_sents = []
            for sent in support_sents:
                processed_support_sents.append(sent)
                json_data = self._extract_financial_metrics_from_json(sent)
                for data in json_data:
                    processed_support_sents.append(data['sentence'])
            # Remove duplicates while preserving order
            seen = set()
            unique_processed = []
            for sent in processed_support_sents:
                if sent not in seen:
                    seen.add(sent)
                    unique_processed.append(sent)
            # Batch encode all claims and all unique processed support sentences
            claim_embs = sbert_model.encode(generic_claims, convert_to_tensor=True, show_progress_bar=False)
            doc_embs = sbert_model.encode(unique_processed, convert_to_tensor=True, show_progress_bar=False)
            sim_matrix = util.cos_sim(claim_embs, doc_embs).cpu().numpy()
            generic_results = []
            for i, claim in enumerate(generic_claims):
                sims = sim_matrix[i]
                topk_idx = sims.argsort()[-self.k:][::-1]
                topk_sents = [(unique_processed[j], float(sims[j])) for j in topk_idx]
                max_sim = float(sims[topk_idx[0]]) if len(topk_idx) > 0 else 0.0
                avg_sim = float(sims[topk_idx].mean()) if len(topk_idx) > 0 else 0.0
                status = "Supported" if avg_sim >= self.threshold else "Potential Hallucination"
                result_entry = {
                    "claim": claim,
                    "claim_type": "generic",
                    "status": status,
                    "max_sim": max_sim,
                    "avg_topk_sim": avg_sim,
                    "topk_sents": topk_sents
                }
                generic_results.append(result_entry)
            # Attach sent and sent_idx from mapping
            for res, mapping in zip(generic_results, generic_mappings):
                res["sent_idx"] = mapping.get("sent_idx")
                res["sent"] = mapping.get("sent")
                results.append(res)
        generic_claims_time = time.time() - start_time
        ic(generic_claims_time)
        start_time = time.time()
        # Step 3: Run entailment on all results (with mapping)
        if results:
            results = check_entailment_domyn_small_with_mapping(results, use_only_entailment)
        entailment_time = time.time() - start_time
        ic(entailment_time)
        
        return results
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sentence_transformers import SentenceTransformer
import spacy
from transformers import pipeline
from tqdm import tqdm
from domynclaimalign.utils.get_hf_token import get_hf_token
from icecream import ic
import re
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI



# model = SentenceTransformer('all-MiniLM-L6-v2')  # Load once

# nlp = spacy.load('en_core_web_sm')

# client = OpenAI(
#     base_url="<Your_Model_API_Endpoint_URL>",
#     api_key="EMPTY"
# )

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
        # From package to examples/FAV
        current_dir.parent.parent.parent / "examples" / "FAV" / "config.json",
        # Alternative path if run from examples
        Path.cwd() / "config.json",
        # Alternative path if run from examples/OWFA/FAV
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

# Initialize global variables with config
model = SentenceTransformer('all-MiniLM-L6-v2')  # Load once
nlp = spacy.load('en_core_web_sm')

# Initialize OpenAI client with config
llm_config = get_llm_client_config()
client = OpenAI(
    base_url=llm_config["base_url"],
    api_key=llm_config["api_key"]
)
MODEL_NAME = llm_config["model_name"]


LIST_OF_FILLER_PHRASES = [
    "Okay", "Sure", "No problem", "Thank", "Wait", "Let me", "Here's", "Here is",
    "As an AI", "I need to", "Here's what I found", "First, I need ", "Breaking it down",
    "To answer your question", "To provide an accurate answer", "To give you the best response",
    "To assist you better", "To help you effectively", "To clarify", "To summarize",
    "In summary", "In conclusion", "Overall", "Generally", "Typically", "Usually",
    "On average", "It's important to note", "It's worth mentioning", 
    "It's essential to understand", "It's crucial to recognize",
    "Looking at the sentence", "Analyzing the sentence", "Examining the sentence",
    "Considering the sentence", "Reflecting on the sentence", "Focusing on the sentence", "\"",
    "So,", "And", "But", "However", "Therefore", "Thus", "Meanwhile", "Additionally",
    "Looking at the text", "(", "[", "{", ">", "<", "...", ")",
]

LIST_OF_PHRASES_TO_EXCLUDE = [
    "I need", "I want", "The user", "the user", "Breaking down the sentence", "text contains",
    "According to the guidelines", "As per the guidelines", "Based on the guidelines",
    "According to the instructions", "As per the instructions", "Based on the instructions",
    "Each fact must be a self-contained", "Do not create partial facts", "Do not omit facts",
    "Do not hallucinate facts", "Facts should be concise yet complete", "If the text contains",
    "Include them in the facts", "Separate each fact with a newline", "No facts found."
]

def normalize_text(text):
    # remove any special characters and extra spaces, and convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return ' '.join(text.lower().strip().split())

def is_fact_supported_embedding(fact, sentence, threshold=0.7):
    embeddings = model.encode([fact, sentence])
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return similarity >= threshold


def load_model_and_tokenizer(model_name="allenai/OLMo-7B", MODEL_CACHE = "../data/model_cache/"):
    # No profiling needed here (cached once)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, add_bos_token=False, add_eos_token=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=MODEL_CACHE)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

def load_sentence_transformer(MODEL_ID_sentence_transformer='all-mpnet-base-v2'):
    return SentenceTransformer(MODEL_ID_sentence_transformer)

def load_spacy(spacy_model="en_core_web_sm"):
    # Load the small English pipeline; in batched calls we disable components we don't use
    return spacy.load(spacy_model)

def is_likely_fact(text, min_words=4):
    sentence_endings = text.count('.') + text.count('?') + text.count('!')
    doc = nlp(text)
    if len(text.split()) < min_words:
        ic("text too short:", text)
        return False
    elif any(text.startswith(phrase) for phrase in LIST_OF_FILLER_PHRASES):
        ic("text starts with filler:", text)
        return False
    elif any(phrase in text for phrase in LIST_OF_PHRASES_TO_EXCLUDE):
        ic("text has unwanted phrase:", text)
        return False
    # elif not text[0].isupper():
    #     ic("text does not start with uppercase:", text)
    #     return False
    # elif not text.strip().endswith(('.', '?', '!')):
    #     ic("text does not end with proper punctuation:", text)
    #     return False
    # # Check for multiple sentences (more than one sentence-ending punctuation)
    
    # elif sentence_endings > 1:
    #     ic("text has multiple sentences:", text)
    #     return False

    elif not any(token.pos_ in ("VERB", "AUX") for token in doc):
        ic("text has no verb or aux:", text)
        return False
    else:   
        return True

def check_the_correctness_of_facts_with_details(text_sents, fact_mappings, batch_size=16):
    """
    Enhanced version that returns both supported facts and failed fact mappings for retry.
    """
    prompts = []
    prompt_to_mapping = {}
    
    for mapping in fact_mappings:
        sent = mapping['sent']
        fact = mapping['fact']
        prompt = f"""Given the sentence: "{sent}", determine if the following fact is a correct atomic fact extracted from it.
    Guidelines:
    - A fact is considered correct if it is an atomic fact of the sentence. (Atomic claims represent a minimal, self-contained proposition within the sentences.)
    - Respond with "No" if the fact is not an atomic fact of the sentence, else respond with "Yes".
    - The fact should not contain reasoning or thoughts of LLMs.
    - The fact should be self-contained and unambiguous and should contain MEANINGFUL facts.
    - The fact should be concise yet complete.
    - The fact should not be multi-sentence.
    - If there is "No facts found.", answer "Yes".
    Sentence: "{sent}"
    Fact: "{fact}"
    Is the fact correct? Answer with "Yes" or "No"."""
        prompts.append(prompt)
        prompt_to_mapping[len(prompts)-1] = mapping

    supported_facts = []
    failed_mappings = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Verifying facts"):
        batch_prompts = prompts[i:i+batch_size]
        for j, prompt in enumerate(batch_prompts):
            result = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a fact verifier."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=64,
                temperature=0.0,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            )
            answer = result.choices[0].message.content.strip().lower()
            mapping = prompt_to_mapping[i + j]
            
            if 'yes' in answer:
                supported_facts.append(mapping['fact'])
            else:
                failed_mappings.append(mapping)

    return supported_facts, failed_mappings

def retry_fact_extraction(failed_mappings, batch_size=8):
    """
    Retry fact extraction for failed mappings with enhanced prompts.
    """
    retry_prompts = []
    mapping_indices = []
    
    for idx, mapping in enumerate(failed_mappings):
        sent = mapping['sent']
        original_fact = mapping['fact']
        
        retry_prompt = f"""The previous fact extraction failed. Please re-extract atomic facts from this sentence more carefully.
    Previous failed fact: "{original_fact}"
    
    Extract all complete atomic facts from the following sentence:
    Guidelines:
    - Each fact must be a self-contained, unambiguous statement that makes sense on its own.
    - Do not create partial facts or incomplete statements.
    - Include every meaningful fact present in the sentence.
    - Facts should be concise yet complete.
    - Include named entities, numbers, metrics, dates, quantities if present.
    - Separate each fact with a newline.
    - Do not include reasoning or filler phrases.
    - If there is no fact, respond with "No facts found."
    
    Sentence: {sent}
    
    Facts:
    """
        retry_prompts.append(retry_prompt)
        mapping_indices.append(idx)
    
    retry_facts = []
    retry_mappings = []
    
    for i in tqdm(range(0, len(retry_prompts), batch_size), desc="Retrying fact extraction"):
        batch_prompts = retry_prompts[i:i+batch_size]
        batch_indices = mapping_indices[i:i+batch_size]
        
        for prompt, orig_idx in zip(batch_prompts, batch_indices):
            result = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an atomic fact extractor. Focus on extracting only valid, complete atomic facts."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=256,
                temperature=0.1,  # Lower temperature for more consistent results
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": True}
                }
            )
            
            extracted_facts = result.choices[0].message.content
            facts = extracted_facts.split('Facts:')[-1].strip().split('\n')
            facts = [fact.strip('- ').strip() for fact in facts if fact.strip()]
            
            # Filter facts using is_likely_fact
            supported = [fact for fact in facts if is_likely_fact(fact)]
            
            original_mapping = failed_mappings[orig_idx]
            for fact in supported:
                retry_mappings.append({
                    "fact": fact,
                    "sent_idx": original_mapping["sent_idx"],
                    "sent": original_mapping["sent"]
                })
                retry_facts.append(fact)
    
    return retry_facts, retry_mappings

def extract_atomic_facts_with_mappings(text, batch_size=8, overlap_threshold=0.15, max_retries=2):
    doc = nlp(text)
    text_sents = [sent.text.strip() for sent in doc.sents if len(sent.text.split()) > 4]
    ic(text_sents)
    
    # Initial extraction
    prompts = [
        f"""Extract all complete atomic facts (smallest unit of information in a sentence) sentences from the following text.
    Guidelines:
    - Do not output any reasoning steps. Do not output the input sentence.
    - Each fact must be a self-contained, unambiguous statement that makes sense on its own.
    - Do not create partial facts. For example, from "Person A was born in place on date", the fact must be "Person A was born in place", not "Person A was born".
    - Include every fact present in the text. Do not omit facts unless they are redundant repeats. Do not hallucinate facts not present in the text.
    - Facts should be concise yet complete. Avoid unnecessary details.
    - Remember that the text may have multiple sentences and special character, so extract facts accordingly.
    - [VERY IMPORTANT] IF THE TEXT CONTAINS NAMED ENTITIES, NUMBERS, METRICS, DATES, QUANTITIES, ETC., INCLUDE THAT INFORMAITON IN THE FACTS. 
    - Separate each fact with a newline.

    Text:
    {text_}

    Facts:
    """ for text_ in text_sents
    ]

    all_facts = []
    fact_mappings = []

    # Initial extraction
    for i in tqdm(range(0, len(prompts), batch_size), desc="Initial fact extraction"):
        batch_prompts = prompts[i:i+batch_size]
        batch_sents = text_sents[i:i+batch_size]
        for sent_idx, (prompt, sent) in enumerate(zip(batch_prompts, batch_sents), start=i):
            result = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an atomic fact extractor."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=256,
                temperature=0.2,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": True}
                }
            )
            extracted_facts = result.choices[0].message.content
            facts = extracted_facts.split('Facts:')[-1].strip().split('\n')
            facts = [fact.strip('- ').strip() for fact in facts if fact.strip()]
            supported = [fact for fact in facts if is_likely_fact(fact)]
            for fact in supported:
                fact_mappings.append({
                    "fact": fact,
                    "sent_idx": sent_idx,
                    "sent": sent
                })
                all_facts.append(fact)

    # Remove duplicates from fact_mappings
    unique_fact_mappings = []
    seen = set()
    for mapping in fact_mappings:
        key = (mapping['fact'], mapping['sent'], mapping['sent_idx'])
        if key not in seen:
            unique_fact_mappings.append(mapping)
            seen.add(key)

    current_mappings = unique_fact_mappings
    final_facts = []
    
    # Verification and retry loop
    for retry_count in range(max_retries):
        ic(f"Verification attempt {retry_count + 1}/{max_retries}")
        
        # Verify current facts
        supported_facts, failed_mappings = check_the_correctness_of_facts_with_details(
            text_sents, current_mappings, batch_size
        )
        
        final_facts.extend(supported_facts)
        ic(f"Supported facts in round {retry_count + 1}: {len(supported_facts)}")
        ic(f"Failed facts in round {retry_count + 1}: {len(failed_mappings)}")
        
        # If no failed facts or last retry, break
        if not failed_mappings or retry_count == max_retries - 1:
            break
            
        # Retry extraction for failed facts
        ic(f"Retrying extraction for {len(failed_mappings)} failed facts")
        retry_facts, retry_mappings = retry_fact_extraction(failed_mappings, batch_size)
        
        # Set current mappings to retry mappings for next verification
        current_mappings = retry_mappings

    # Remove duplicates from final facts
    final_facts = list(set(final_facts))
    ic(f"Final facts count: {len(final_facts)}")
    ic(final_facts)
    
    # Create final mappings for supported facts only
    final_mappings = [mapping for mapping in unique_fact_mappings + 
                     sum([retry_mappings for _, retry_mappings in 
                         [retry_fact_extraction(failed_mappings, batch_size) 
                          for failed_mappings in [[]]]], [])
                     if mapping['fact'] in final_facts]
    
    return {
        "all_facts": final_facts,
        "fact_mappings": final_mappings
    }
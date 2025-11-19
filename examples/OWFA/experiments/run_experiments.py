import sys
import os
import time
import argparse
import json
import hashlib
from datetime import datetime
import spacy
from icecream import ic
sys.path.append('..')
from sentence_transformers import SentenceTransformer
from pathlib import Path
import main.answer_generation as answer_generation
from domynclaimalign.utils.model_utils import extract_atomic_facts_with_mappings
from domynclaimalign.main.compute_traces import compute_traces
from domynclaimalign.main.hallucination_claim_support import HallucinationClaimSupport_advanced_withMapping

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuration constants
MODEL_ID_sentence_transformer = 'all-mpnet-base-v2'
WIKI_BASE_DIR = "../data/wiki"
WIKI_INDEX_DIR = "../data/wiki_index"
UNIGRAM_PATH = "../data/wiki_unigram_probs/olmo-7b_wiki_unigram_probs.json"
MODEL_ID = "allenai/OLMo-7B"
MODEL_CACHE = "../data/model_cache/"
SPACY_MODEL = "en_core_web_sm"

# Load models with error handling
try:
    sbert_model = SentenceTransformer(MODEL_ID_sentence_transformer)
    print(f"✓ Loaded SentenceTransformer: {MODEL_ID_sentence_transformer}")
except Exception as e:
    print(f"✗ Failed to load SentenceTransformer: {e}")
    sbert_model = None

try:
    HallucinationClaimSupport_obj = HallucinationClaimSupport_advanced_withMapping(threshold=0.6)
    print(f"✓ Loaded HallucinationClaimSupport")
except Exception as e:
    print(f"✗ Failed to load HallucinationClaimSupport: {e}")
    HallucinationClaimSupport_obj = None

try:
    nlp = spacy.load(SPACY_MODEL)
    print(f"✓ Loaded spaCy model: {SPACY_MODEL}")
except OSError as e:
    print(f"✗ Failed to load spaCy model {SPACY_MODEL}: {e}")
    nlp = None

def generate_prompt_hash(prompt):
    """Generate a hash for the prompt to ensure uniqueness"""
    return hashlib.md5(prompt.encode()).hexdigest()

def run_experiment_on_prompt(prompt_obj, experiment_id, run_id, experiment_name):
    """Run a single experiment on a prompt and return detailed results"""
    try:
        prompt = prompt_obj["prompt"]
        start_time = time.time()
        
        # Initialize timing components
        answer_generation_start = time.time()
        
        # Generate answer with error handling
        try:
            answer = answer_generation.generate_answer(prompt, MODEL_ID, MODEL_CACHE)
            ic(f"Generated answer length: {len(answer) if answer else 0}")
            answer_generation_runtime = time.time() - answer_generation_start
            answer_generation_success = True
        except Exception as e:
            print(f"    ✗ Answer generation failed: {str(e)}")
            answer = f"Error generating answer: {str(e)}"
            answer_generation_runtime = time.time() - answer_generation_start
            answer_generation_success = False

        # Compute traces with error handling
        trace_start = time.time()
        try:
            restricted_docs_query, restricted_docs_answer, span_docs, trace_error, timings, entities = compute_traces(
                prompt, answer, MODEL_ID, MODEL_CACHE, UNIGRAM_PATH, SPACY_MODEL, 
                MODEL_ID_sentence_transformer, WIKI_BASE_DIR, WIKI_INDEX_DIR
            )
            trace_success = trace_error is None
            trace_runtime = time.time() - trace_start
            ic(len(restricted_docs_query), len(restricted_docs_answer), len(span_docs))
        except Exception as e:
            print(f"    ✗ Trace computation failed: {str(e)}")
            restricted_docs_query, restricted_docs_answer, span_docs = [], [], []
            trace_error = str(e)
            trace_success = False
            trace_runtime = time.time() - trace_start
            timings = {}
            entities = []

        matched_docs = restricted_docs_query + restricted_docs_answer + span_docs
        retrieval_success = len(matched_docs) > 0
        entity_success = len(entities) > 0 if entities else True  # Success if entities extracted or not required
        
        # Extract atomic facts with error handling
        fact_start = time.time()
        try:
            facts_result = extract_atomic_facts_with_mappings(answer)
            all_facts = facts_result.get('all_facts', [])
            fact_mappings = facts_result.get('fact_mappings', [])
            fact_success = True
            ic(f"Extracted {len(all_facts)} facts and {len(fact_mappings)} mappings")
        except Exception as e:
            print(f"    ✗ Fact extraction failed: {str(e)}")
            all_facts = []
            fact_mappings = []
            fact_success = False

        fact_runtime = time.time() - fact_start
        
        # Hallucination claim support with error handling
        halluc_start = time.time()
        # try:
        if HallucinationClaimSupport_obj is None:
            raise Exception("HallucinationClaimSupport object not loaded")
        if sbert_model is None:
            raise Exception("SentenceTransformer model not loaded")
        
        n = len(matched_docs)
        highly_relevant_docs = matched_docs[:max(1, n//5)] if n > 0 else []

        hallucination_results = HallucinationClaimSupport_obj.check_claims(
            fact_mappings, highly_relevant_docs, sbert_model, 
            use_numerical=False, use_only_entailment=False
        )
        ic(f"Analyzed {len(hallucination_results)} claims for hallucination")
        halluc_success = True
        # except Exception as e:
        #     print(f"    ✗ Hallucination analysis failed: {str(e)}")
        #     hallucination_results = []
        #     halluc_success = False
            
        halluc_runtime = time.time() - halluc_start

        """hallucination_results are a list of dictionaries containing information about each claim and its support status.
                {'claim': entry['claim'],
                'sent_idx': entry.get('sent_idx'),
                'sent': entry.get('sent'),
                'status': entry.get('status', ''),
                'max_sim': entry.get('max_sim', 0.0),
                'avg_topk_sim': entry.get('avg_topk_sim', 0.0),
                'topk_sents': topk,
                'entailment_score': entailment_score
            }) """

        # Calculate metrics with error handling
        try:
            supported_claims = sum(1 for result in hallucination_results if result.get('status', '').lower().startswith('support'))
            unsupported_claims = sum(1 for result in hallucination_results if result.get('status', '').lower().startswith('potential'))
            claims_analyzed = supported_claims + unsupported_claims
            support_rate = supported_claims / claims_analyzed if claims_analyzed > 0 else 0.0
            sentences_with_facts = len([result for result in hallucination_results if result.get('sent')])
        except Exception as e:
            print(f"    ✗ Metrics calculation failed: {str(e)}")
            supported_claims = unsupported_claims = claims_analyzed = sentences_with_facts = 0
            support_rate = 0.0

        total_runtime = time.time() - start_time
        
        # Build result structure matching the expected format
        result = {
            "experiment_metadata": {
                "experiment_id": experiment_id,
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "prompt_data": {
                    "id": prompt_obj.get("id", experiment_id),
                    "query_category": prompt_obj.get("category", ""),
                    "query_subcategory": prompt_obj.get("subcategory", ""),
                    "topic": prompt_obj.get("topic", ""),
                    "template": prompt_obj.get("template", ""),
                    "prompt": prompt,
                },
                "prompt_hash": generate_prompt_hash(prompt),
            },
            "system_performance": {
                "total_runtime": total_runtime,
                "answer_generation_runtime": answer_generation_runtime,
                "alignment_consistency_scoring_runtime": timings.get('alignment_consistency_scoring', 0.0),
                "bm25_scoring_sorting_runtime": timings.get('bm25_scoring_sorting', 0.0),
                "document_retrieval_merging_runtime": timings.get('document_retrieval_merging', 0.0),
                "entity_based_doc_filtering_runtime": timings.get('entity_based_doc_filtering', 0.0),
                "entity_extraction_runtime": timings.get('entity_extraction', 0.0),
                "loading_resources_runtime": timings.get('loading_resources', 0.0),
                "maximal_span_matching_runtime": timings.get('maximal_span_matching', 0.0),
                "tokenize_answer_runtime": timings.get('tokenize_answer', 0.0),
                "fact_runtime": fact_runtime,
                "halluc_runtime": halluc_runtime,
                "trace_success": trace_success,
                "entity_success": entity_success,
                "retrieval_success": retrieval_success,
                "fact_success": fact_success,
                "halluc_success": halluc_success
            },
            "outputs": {
                "answer": answer,
                "entities": entities,
                "matched_docs_count": len(matched_docs),
                "facts_count": claims_analyzed,
                "claims_analyzed": claims_analyzed,
                "supported_claims": supported_claims,
                "unsupported_claims": unsupported_claims,
                "support_rate": support_rate,
                "sentences_with_facts": sentences_with_facts
            },
            "raw_data": {
                "matched_docs": [
                    [doc.get('doc', {}).get('text', '')[:100], doc.get('alignment_score', 0)]
                    for doc in matched_docs
                ],
                "fact_mappings": fact_mappings,
                "hallucination_results": hallucination_results,
            }
        }
        
        return result
    
    except Exception as e:
        # Fallback error handling for entire function
        print(f"    ✗ Critical error in experiment: {str(e)}")
        return {
            "experiment_metadata": {
                "experiment_id": experiment_id,
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "prompt_data": {
                    "id": prompt_obj.get("id", experiment_id),
                    "prompt": prompt_obj.get("prompt", "ERROR: Could not access prompt"),
                },
                "prompt_hash": generate_prompt_hash(prompt_obj.get("prompt", "error")),
            },
            "system_performance": {
                "total_runtime": 0.0,
                "answer_generation_runtime": 0.0,
                "fact_runtime": 0.0,
                "halluc_runtime": 0.0,
                "trace_success": False,
                "entity_success": False,
                "retrieval_success": False,
                "fact_success": False,
                "halluc_success": False,
                "error": str(e)
            },
            "outputs": {
                "answer": f"Error: {str(e)}",
                "entities": [],
                "matched_docs_count": 0,
                "facts_count": 0,
                "claims_analyzed": 0,
                "supported_claims": 0,
                "unsupported_claims": 0,
                "support_rate": 0.0,
                "sentences_with_facts": 0
            },
            "raw_data": {
                "matched_docs": [],
                "fact_mappings": [],
                "hallucination_results": [],
            }
        }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run comprehensive agent system experiments')
    # parser.add_argument('--prompts-file', default='experiment_data/prompt_datasets/evaluation_prompts_wikipedia_topics.json', 
    #                    help='Path to prompts JSON file')
    parser.add_argument('--prompts-file', default='experiment_data/prompt_datasets/fact_check_prompts_wikipedia_topics.json', 
                       help='Path to prompts JSON file')
    parser.add_argument('--experiment-name', default='pilot_experiment', 
                       help='Name for this experiment batch')
    parser.add_argument('--max-prompts', type=int, default=2, 
                       help='Maximum number of prompts to test (for quick testing)')
    parser.add_argument('--runs-per-prompt', type=int, default=2, 
                       help='Number of runs per prompt')
    parser.add_argument('--full', action='store_true', 
                       help='Run full experiment (ignores max-prompts)')
    
    args = parser.parse_args()
    
    # Validate essential components before starting
    print("Validating system components...")
    validation_errors = []
    
    if sbert_model is None:
        validation_errors.append("SentenceTransformer model not loaded")
    if HallucinationClaimSupport_obj is None:
        validation_errors.append("HallucinationClaimSupport object not loaded")
    if nlp is None:
        validation_errors.append("spaCy model not loaded")
        
    # Check if data directories exist
    if not os.path.exists(WIKI_BASE_DIR):
        validation_errors.append(f"Wiki base directory not found: {WIKI_BASE_DIR}")
    if not os.path.exists(WIKI_INDEX_DIR):
        validation_errors.append(f"Wiki index directory not found: {WIKI_INDEX_DIR}")
    if not os.path.exists(UNIGRAM_PATH):
        validation_errors.append(f"Unigram probabilities file not found: {UNIGRAM_PATH}")
    if not os.path.exists(MODEL_CACHE):
        validation_errors.append(f"Model cache directory not found: {MODEL_CACHE}")
    
    if validation_errors:
        print("✗ Validation failed with the following errors:")
        for error in validation_errors:
            print(f"  - {error}")
        print("\nNote: Some components may still work with missing dependencies.")
        print("Continuing with experiment (errors will be handled gracefully)...")
    else:
        print("✓ All components validated successfully")
    
    # Load prompts
    try:
        with open(args.prompts_file, "r") as f:
            prompts = json.load(f)
    except FileNotFoundError:
        print(f"Error: Prompts file not found: {args.prompts_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in prompts file: {args.prompts_file}")
        return
    
    # Determine how many prompts to run
    if args.full:
        prompts_to_run = prompts
    else:
        prompts_to_run = prompts[:args.max_prompts]
    
    print(f"Starting experiment: {args.experiment_name}")
    print(f"Running {len(prompts_to_run)} prompts with {args.runs_per_prompt} runs each")
    print(f"Total experiments: {len(prompts_to_run) * args.runs_per_prompt}")
    
    # Create output directory with error handling
    out_dir = "experiment_data/experiment_results"
    try:
        os.makedirs(out_dir, exist_ok=True)
        print(f"✓ Output directory created/verified: {out_dir}")
    except Exception as e:
        print(f"✗ Failed to create output directory {out_dir}: {str(e)}")
        return
    
    # Track batch metadata
    batch_start_time = time.time()
    successful_experiments = 0
    all_prompt_results = []
    
    # Run experiments
    for prompt_idx, prompt_obj in enumerate(prompts_to_run):
        print(f"Running prompt {prompt_idx + 1}/{len(prompts_to_run)}: {prompt_obj.get('prompt', 'No prompt text')}")
        
        prompt_runs = []
        
        for run_id in range(args.runs_per_prompt):
            print(f"  Run {run_id + 1}/{args.runs_per_prompt}")
            try:
                result = run_experiment_on_prompt(prompt_obj, prompt_idx, run_id, args.experiment_name)
                prompt_runs.append(result)
                successful_experiments += 1
                print(f"    ✓ Completed successfully")
            except Exception as e:
                print(f"    ✗ Failed: {str(e)}")
                # Create a failure result
                failure_result = {
                    "experiment_metadata": {
                        "experiment_id": prompt_idx,
                        "run_id": run_id,
                        "timestamp": datetime.now().isoformat(),
                        "prompt_data": {
                            "id": prompt_obj.get("id", prompt_idx),
                            "prompt": prompt_obj.get("prompt", ""),
                        },
                        "prompt_hash": generate_prompt_hash(prompt_obj.get("prompt", "")),
                    },
                    "system_performance": {
                        "total_runtime": 0.0,
                        "answer_generation_runtime": 0.0,
                        "fact_runtime": 0.0,
                        "halluc_runtime": 0.0,
                        "trace_success": False,
                        "entity_success": False,
                        "retrieval_success": False,
                        "fact_success": False,
                        "halluc_success": False,
                        "error": str(e)
                    },
                    "outputs": {
                        "answer": f"Error: {str(e)}",
                        "entities": [],
                        "matched_docs_count": 0,
                        "facts_count": 0,
                        "claims_analyzed": 0,
                        "supported_claims": 0,
                        "unsupported_claims": 0,
                        "support_rate": 0.0,
                        "sentences_with_facts": 0
                    },
                    "raw_data": {
                        "matched_docs": [],
                        "fact_mappings": [],
                        "hallucination_results": [],
                    }
                }
                prompt_runs.append(failure_result)

        # Save individual prompt results with error handling
        prompt_output_file = os.path.join(out_dir, f"{args.experiment_name}_prompt_{prompt_idx}_all_runs.json")
        try:
            with open(prompt_output_file, "w") as f:
                json.dump(prompt_runs, f, indent=2)
            print(f"  Saved results to {prompt_output_file}")
        except Exception as e:
            print(f"  ✗ Failed to save results to {prompt_output_file}: {str(e)}")
        
        all_prompt_results.extend(prompt_runs)
    
    batch_end_time = time.time()
    
    # Create batch metadata
    batch_metadata = {
        "experiment_name": args.experiment_name,
        "total_prompts": len(prompts_to_run),
        "runs_per_prompt": args.runs_per_prompt,
        "total_experiments": len(prompts_to_run) * args.runs_per_prompt,
        "successful_experiments": successful_experiments,
        "batch_start_time": batch_start_time,
        "batch_end_time": batch_end_time,
        "total_batch_time": batch_end_time - batch_start_time
    }
    
    # Save batch metadata with error handling
    metadata_file = os.path.join(out_dir, f"{args.experiment_name}_metadata.json")
    try:
        with open(metadata_file, "w") as f:
            json.dump(batch_metadata, f, indent=2)
        print(f"Batch metadata saved to: {metadata_file}")
    except Exception as e:
        print(f"✗ Failed to save batch metadata to {metadata_file}: {str(e)}")
    
    print(f"\nExperiment batch completed!")
    print(f"Total runtime: {batch_metadata['total_batch_time']:.2f} seconds")
    print(f"Successful experiments: {successful_experiments}/{batch_metadata['total_experiments']}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Experiment interrupted by user (Ctrl+C)")
        print("Partial results may have been saved.")
    except Exception as e:
        print(f"\n\n✗ Critical error in main execution: {str(e)}")
        print("Please check your configuration and try again.")
        import traceback
        print("\nFull error traceback:")
        traceback.print_exc()
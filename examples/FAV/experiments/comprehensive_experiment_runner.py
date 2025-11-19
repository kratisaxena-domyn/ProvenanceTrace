import json
import time
import os
import sys
import shutil
from datetime import datetime
from icecream import ic
import hashlib

sys.path.append('..')
from main.agent_system_complex import ReActOrchestrator
from main.show_traces import extract_turn_trace, get_entities_keywords_from_answer, remove_duplicates
from domynclaimalign.utils.model_utils import extract_atomic_facts_with_mappings
from domynclaimalign.utils.agent_create_index import Normalize_Index_Retrieve
from domynclaimalign.main.hallucination_claim_support import HallucinationClaimSupport_advanced_withMapping
from sentence_transformers import SentenceTransformer
import spacy

class ExperimentRunner:
    def __init__(self):
        # Initialize models and components
        self.sbert_model = SentenceTransformer('all-mpnet-base-v2')
        self.normalize_retrieve = Normalize_Index_Retrieve()
        self.halluc_checker = HallucinationClaimSupport_advanced_withMapping(threshold=0.5)
        
        # Load spaCy model for sentence segmentation
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.nlp = spacy.load("en_core_web_sm")
        
        # Create experiment directories
        self.experiment_base_dir = "experiment_data"
        self.agent_data_dir = os.path.join(self.experiment_base_dir, "agent_data")
        self.results_dir = os.path.join(self.experiment_base_dir, "experiment_result")
        
        os.makedirs(self.agent_data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Track processed prompts to avoid duplicate API calls
        self.processed_prompts = {}
        self.rate_limit_delay = 0.2  # 5 requests per second = 0.2 seconds between requests
        self.last_api_call_time = 0
    
    def get_prompt_hash(self, prompt_text, tickers):
        """Generate a hash for the prompt to identify duplicates"""
        content = f"{prompt_text}_{sorted(tickers)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def respect_rate_limit(self):
        """Ensure we don't exceed 5 requests per second"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call_time
        if time_since_last_call < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_call
            time.sleep(sleep_time)
        self.last_api_call_time = time.time()
    
    def save_agent_logs(self, experiment_id, run_id, source_log_dir="../agent_logs"):
        """Save agent logs for the experiment"""
        experiment_log_dir = os.path.join(self.agent_data_dir, f"exp_{experiment_id}_run_{run_id}")
        os.makedirs(experiment_log_dir, exist_ok=True)
        
        # Copy all log files from source to experiment directory
        if os.path.exists(source_log_dir):
            for filename in os.listdir(source_log_dir):
                if filename.endswith('.jsonl'):
                    src_path = os.path.join(source_log_dir, filename)
                    dst_path = os.path.join(experiment_log_dir, filename)
                    shutil.copy2(src_path, dst_path)
        
        return experiment_log_dir
    
    def run_single_experiment(self, prompt_data, experiment_id, run_id):
        """Run single prompt through the system and collect all metrics"""
        start_time = time.time()
        
        # Check if we've already processed this exact prompt
        prompt_hash = self.get_prompt_hash(prompt_data["prompt"], prompt_data["tickers"])
        
        # Respect rate limits before making any API calls
        self.respect_rate_limit()
        
        # --- Clear the log directory here ---
        source_log_dir = "../agent_logs"
        if os.path.exists(source_log_dir):
            for filename in os.listdir(source_log_dir):
                if filename.endswith('.jsonl'):
                    os.remove(os.path.join(source_log_dir, filename))
                    
        # Create fresh orchestrator for each experiment to avoid state contamination
        orchestrator = ReActOrchestrator()
        
        # Run the agent system
        agent_start_time = time.time()
        # try:
        answer = orchestrator.run(prompt_data["prompt"])
        agent_success = True
        # except Exception as e:
        #     answer = f"Error: {str(e)}"
        #     agent_success = False
        #     ic(f"Agent error for experiment {experiment_id}, run {run_id}: {str(e)}")
        agent_runtime = time.time() - agent_start_time
        
        # Save agent logs immediately after the run
        experiment_log_dir = self.save_agent_logs(experiment_id, run_id)
        
        # Extract turn trace using the saved logs
        turn_number = orchestrator.turn_number - 1
        AGENT_LOG_FILES = [
            'orchestrator_log.jsonl',
            'InternetAgent_log.jsonl', 
            'WebSearchAgent_log.jsonl',
            'YHFinanceAPIAgent_log.jsonl',
            'JSONExtractorAgent_log.jsonl',
            'SummarizerAgent_log.jsonl'
        ]
        
        trace_start_time = time.time()
        try:
            trace_result = extract_turn_trace(turn_number, experiment_log_dir, AGENT_LOG_FILES)
            trace_success = True
        except Exception as e:
            trace_result = {"steps": [], "answer": answer}
            trace_success = False
            ic(f"Trace extraction error: {str(e)}")
        trace_runtime = time.time() - trace_start_time
        
        # Get answer from trace or use orchestrator answer
        answer_from_trace = trace_result.get("answer", "")
        if not answer_from_trace:
            answer_from_trace = answer
        
        trail = trace_result.get("steps", [])
        
        # Remove duplicates from trail
        dedup_start_time = time.time()
        trail = remove_duplicates(trail)
        dedup_runtime = time.time() - dedup_start_time
        
        # Entity extraction
        entity_start_time = time.time()
        try:
            entities_keywords = get_entities_keywords_from_answer(answer_from_trace)
            entities = [ek[0] for ek in entities_keywords.get("entities", [])]
            entity_success = True
        except Exception as e:
            entities = []
            entity_success = False
            ic(f"Entity extraction error: {str(e)}")
        entity_runtime = time.time() - entity_start_time
        
        # Document retrieval
        retrieval_start_time = time.time()
        try:
            matched_docs = self.normalize_retrieve.return_matching_docs(
                entities, trail, answer_from_trace, self.sbert_model
            ) if entities else []
            retrieval_success = True
        except Exception as e:
            matched_docs = []
            retrieval_success = False
            ic(f"Document retrieval error: {str(e)}")
        retrieval_runtime = time.time() - retrieval_start_time
        
        # Fact extraction - only if we have a valid answer
        fact_start_time = time.time()
        if answer_from_trace and answer_from_trace.strip() and not answer_from_trace.startswith("Error:"):
            try:
                facts_result = extract_atomic_facts_with_mappings(answer_from_trace)
                fact_mappings = facts_result.get('fact_mappings', [])
                all_facts = facts_result.get('all_facts', [])
                fact_success = True
            except Exception as e:
                fact_mappings = []
                all_facts = []
                fact_success = False
                ic(f"Fact extraction error: {str(e)}")
        else:
            fact_mappings = []
            all_facts = []
            fact_success = True  # No facts to extract is not a failure
        fact_runtime = time.time() - fact_start_time
        
        # Hallucination detection - only if we have facts and matched docs
        halluc_start_time = time.time()
        if fact_mappings and matched_docs:
            try:
                hallucination_results = self.halluc_checker.check_claims(
                    fact_mappings, matched_docs, self.sbert_model
                )
                halluc_success = True
            except Exception as e:
                hallucination_results = []
                halluc_success = False
                ic(f"Hallucination detection error: {str(e)}")
        else:
            hallucination_results = []
            halluc_success = True  # No facts/docs to check is not a failure
        halluc_runtime = time.time() - halluc_start_time
        
        # Map facts to sentences for analysis
        mapping_start_time = time.time()
        facts_by_sentence = []
        if answer_from_trace and fact_mappings:
            try:
                doc = self.nlp(answer_from_trace)
                text_sents = [sent.text.strip() for sent in doc.sents if len(sent.text.split()) > 4]
                facts_by_sentence = [[] for _ in text_sents]
                
                for mapping in fact_mappings:
                    sent_idx = mapping.get('sent_idx', 0)
                    if 0 <= sent_idx < len(facts_by_sentence):
                        facts_by_sentence[sent_idx].append(mapping)
            except Exception as e:
                ic(f"Sentence mapping error: {str(e)}")
        mapping_runtime = time.time() - mapping_start_time
        
        # Calculate claim support statistics
        support_stats_start = time.time()
        supported_claims = 0
        unsupported_claims = 0
        total_claims = len(hallucination_results)
        
        for hr in hallucination_results:
            status = hr.get('status', '').lower()
            if status.startswith('Support'):
                supported_claims += 1
            else:
                unsupported_claims += 1
        
        support_rate = supported_claims / total_claims if total_claims > 0 else 0
        support_stats_runtime = time.time() - support_stats_start
        
        total_runtime = time.time() - start_time
        
        # Compile comprehensive results
        result = {
            "experiment_metadata": {
                "experiment_id": experiment_id,
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "prompt_data": prompt_data,
                "prompt_hash": prompt_hash,
                "agent_log_dir": experiment_log_dir
            },
            "system_performance": {
                "total_runtime": total_runtime,
                "agent_runtime": agent_runtime,
                "trace_runtime": trace_runtime,
                "dedup_runtime": dedup_runtime,
                "entity_runtime": entity_runtime,
                "retrieval_runtime": retrieval_runtime,
                "fact_runtime": fact_runtime,
                "halluc_runtime": halluc_runtime,
                "mapping_runtime": mapping_runtime,
                "support_stats_runtime": support_stats_runtime,
                "agent_success": agent_success,
                "trace_success": trace_success,
                "entity_success": entity_success,
                "retrieval_success": retrieval_success,
                "fact_success": fact_success,
                "halluc_success": halluc_success
            },
            "outputs": {
                "answer": answer_from_trace,
                "entities": entities,
                "matched_docs_count": len(matched_docs),
                "facts_count": len(fact_mappings),
                "claims_analyzed": total_claims,
                "supported_claims": supported_claims,
                "unsupported_claims": unsupported_claims,
                "support_rate": support_rate,
                "sentences_with_facts": len([s for s in facts_by_sentence if s])
            },
            "raw_data": {
                "trace_steps": trail,
                "matched_docs": matched_docs,
                "fact_mappings": fact_mappings,
                "all_facts": all_facts,
                "hallucination_results": hallucination_results,
                "facts_by_sentence": facts_by_sentence,
                "entities_keywords": entities_keywords if 'entities_keywords' in locals() else {}
            }
        }
        
        return result
    
    def run_experiment_batch(self, prompts, experiment_name, runs_per_prompt=3):
        """Run batch of experiments with multiple runs per prompt"""
        batch_results = []
        batch_start_time = time.time()
        
        print(f"Starting experiment batch '{experiment_name}' with {len(prompts)} prompts, {runs_per_prompt} runs each")
        print(f"Total experiments to run: {len(prompts) * runs_per_prompt}")
        
        for i, prompt_data in enumerate(prompts):
            prompt_start_time = time.time()
            print(f"\nRunning prompt {i+1}/{len(prompts)}: {prompt_data['prompt'][:80]}...")
            
            prompt_results = []
            for run in range(runs_per_prompt):
                run_start_time = time.time()
                print(f"  Run {run+1}/{runs_per_prompt}...", end=" ")
                
                try:
                    result = self.run_single_experiment(prompt_data, i, run)
                    prompt_results.append(result)
                    batch_results.append(result)
                    
                    # Save individual result
                    # result_file = os.path.join(self.results_dir, f"{experiment_name}_prompt_{i}_run_{run}.json")
                    # with open(result_file, "w") as f:
                    #     json.dump(result, f, indent=2, default=str)
                    
                    run_time = time.time() - run_start_time
                    print(f"✓ ({run_time:.1f}s)")
                    
                except Exception as e:
                    print(f"✗ Error: {str(e)}")
                    ic(f"Experiment failed - prompt {i}, run {run}: {str(e)}")
                    continue
            
            prompt_time = time.time() - prompt_start_time
            print(f"  Prompt completed in {prompt_time:.1f}s")
            
            # Save prompt batch results
            prompt_batch_file = os.path.join(self.results_dir, f"{experiment_name}_prompt_{i}_all_runs.json")
            with open(prompt_batch_file, "w") as f:
                json.dump(prompt_results, f, indent=2, default=str)
        
        # Save complete batch summary
        # batch_file = os.path.join(self.results_dir, f"{experiment_name}_batch_results.json")
        # with open(batch_file, "w") as f:
        #     json.dump(batch_results, f, indent=2, default=str)
        
        # Save batch metadata
        batch_metadata = {
            "experiment_name": experiment_name,
            "total_prompts": len(prompts),
            "runs_per_prompt": runs_per_prompt,
            "total_experiments": len(batch_results),
            "successful_experiments": len([r for r in batch_results if r["system_performance"]["agent_success"]]),
            "batch_start_time": batch_start_time,
            "batch_end_time": time.time(),
            "total_batch_time": time.time() - batch_start_time
        }
        
        metadata_file = os.path.join(self.results_dir, f"{experiment_name}_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(batch_metadata, f, indent=2, default=str)
        
        print(f"\nBatch '{experiment_name}' completed!")
        print(f"Total time: {batch_metadata['total_batch_time']:.1f}s")
        print(f"Successful experiments: {batch_metadata['successful_experiments']}/{batch_metadata['total_experiments']}")
        
        return batch_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comprehensive agent system experiments')
    parser.add_argument('--prompts-file', default='experiment_data/prompt_datasets/evaluation_prompts_with_categories_single_ticker.json', 
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
    
    # Load prompts
    try:
        with open(args.prompts_file, "r") as f:
            prompts = json.load(f)
        print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
    except FileNotFoundError:
        print(f"Error: Prompts file {args.prompts_file} not found")
        print("Please run create_prompt_dataset.py first to generate prompts")
        sys.exit(1)
    
    # Select prompts to run
    if args.full:
        test_prompts = prompts
        print(f"Running FULL experiment with all {len(prompts)} prompts")
    else:
        test_prompts = prompts[:args.max_prompts]
        print(f"Running pilot experiment with first {len(test_prompts)} prompts")
    
    # Initialize and run experiments
    runner = ExperimentRunner()
    results = runner.run_experiment_batch(
        test_prompts, 
        args.experiment_name, 
        runs_per_prompt=args.runs_per_prompt
    )
    
    print(f"\nExperiment complete! Results saved in {runner.results_dir}")
    # print(f"Agent logs saved in {runner.agent_data_dir}")
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from icecream import ic
import time
import concurrent.futures
from metrics_calculation_utils.llm_evaluations import LLM_Evaluations
LLM_Evaluations_obj = LLM_Evaluations()

class SentenceMatchingCalculator:
    def __init__(self, results, plots_dir):
        self.results = results
        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)

    def evaluate_sentence_matching(self):
        """Evaluate sentence-level claim-evidence matching using LLM-as-judge"""
        t0 = time.time()
        print("Evaluating sentence-level claim-evidence matching...")
        
        sentence_results = []
        
        def judge_sentence_support(args):
            claim, sentence, original_prompt = args
            return LLM_Evaluations_obj._llm_judge_sentence_support(claim, sentence, original_prompt)
        
        for i, result in tqdm(enumerate(self.results), total=len(self.results)):
            if i % 5 == 0:
                print(f"Processing result {i+1}/{len(self.results)}")
            
            fact_mappings = result["raw_data"]["fact_mappings"]
            matched_docs = result["raw_data"]["matched_docs"]
            original_prompt = result["experiment_metadata"]["prompt_data"]["prompt"]
            
            if not fact_mappings or not matched_docs:
                continue
            
            for fact_mapping in tqdm(fact_mappings):
                claim = fact_mapping.get('fact', '')
                if not claim:
                    continue
                
                # Extract sentences from documents
                sentences = []
                for doc in matched_docs:
                    if isinstance(doc, dict):
                        doc_content = doc.get('content', str(doc))
                    elif isinstance(doc, list):
                        doc_content = '\n'.join([str(item) for item in doc])
                    else:
                        doc_content = str(doc)
                    
                    # Simple sentence splitting
                    doc_sentences = [s.strip() for s in doc_content.split('.') if len(s.strip()) > 20]
                    sentences.extend(doc_sentences)
                
                if not sentences:
                    continue
                
                # Parallel evaluation of sentences
                args_list = [(claim, sentence, original_prompt) for sentence in sentences] 
                with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
                    sentence_evals = list(executor.map(judge_sentence_support, args_list))
                
                supporting_sentences = [eval_result for eval_result in sentence_evals if eval_result["supports_claim"]]
                
                sentence_results.append({
                    "experiment_id": result["experiment_metadata"]["experiment_id"],
                    "claim": claim,
                    "total_sentences_evaluated": len(sentence_evals),
                    "supporting_sentences": len(supporting_sentences),
                    "precision": len(supporting_sentences) / len(sentence_evals) if sentence_evals else 0,
                    "avg_support_strength": np.mean([s["support_strength"] for s in supporting_sentences]) if supporting_sentences else 0,
                    "sentence_evaluations": sentence_evals
                })
        
        # Calculate aggregate metrics
        metrics = {
            "avg_sentence_precision": np.mean([r["precision"] for r in sentence_results]),
            "avg_support_strength": np.mean([r["avg_support_strength"] for r in sentence_results]),
            "total_claims_evaluated": len(sentence_results),
            "total_sentences_evaluated": sum([r["total_sentences_evaluated"] for r in sentence_results])
        }
        
        # Plot sentence matching analysis
        self._plot_sentence_matching_analysis(sentence_results, metrics)
        
        # Profiling
        sentence_matching_time = time.time() - t0
        ic(sentence_matching_time)
        
        return metrics, sentence_results

    def _plot_sentence_matching_analysis(self, sentence_results, metrics):
        """Plot sentence matching analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Sentence precision distribution
        precisions = [r["precision"] for r in sentence_results]
        axes[0, 0].hist(precisions, bins=15, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Sentence Matching Precision Distribution')
        axes[0, 0].set_xlabel('Precision')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(metrics["avg_sentence_precision"], color='red', linestyle='--',
                          label=f'Mean: {metrics["avg_sentence_precision"]:.3f}')
        axes[0, 0].legend()
        
        # Support strength distribution
        support_strengths = [r["avg_support_strength"] for r in sentence_results]
        axes[0, 1].hist(support_strengths, bins=15, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Average Support Strength Distribution')
        axes[0, 1].set_xlabel('Support Strength')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(metrics["avg_support_strength"], color='red', linestyle='--',
                          label=f'Mean: {metrics["avg_support_strength"]:.3f}')
        axes[0, 1].legend()
        
        # Precision vs Support Strength scatter
        axes[1, 0].scatter(precisions, support_strengths, alpha=0.6)
        axes[1, 0].set_title('Precision vs Support Strength')
        axes[1, 0].set_xlabel('Sentence Precision')
        axes[1, 0].set_ylabel('Average Support Strength')
        
        # Supporting sentences count distribution
        supporting_counts = [r["supporting_sentences"] for r in sentence_results]
        axes[1, 1].hist(supporting_counts, bins=range(0, max(supporting_counts) + 2), 
                       alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Supporting Sentences Found per Claim')
        axes[1, 1].set_xlabel('Number of Supporting Sentences')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'sentence_matching_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
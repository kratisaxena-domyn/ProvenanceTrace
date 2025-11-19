import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import concurrent.futures
from icecream import ic
from metrics_calculation_utils.llm_evaluations import LLM_Evaluations

LLM_Evaluations_obj = LLM_Evaluations()

class CoverageCalculator:
    def __init__(self, results, plots_dir):
        self.results = results
        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)

    def evaluate_coverage_rate(self):
        """Evaluate coverage rate - simply check if claims have supporting sentences"""
        t0 = time.time()
        print("Evaluating coverage rate...")
        
        coverage_results = []
        
        def evaluate_claim_coverage(args):
            claim, matched_docs, original_prompt = args
            return self._evaluate_single_claim_coverage(claim, matched_docs, original_prompt)
        
        for i, result in tqdm(enumerate(self.results), total=len(self.results)):
            if i % 5 == 0:
                print(f"Processing result {i+1}/{len(self.results)}")

            fact_mappings = result["raw_data"]["fact_mappings"]
            matched_docs = result["raw_data"]["matched_docs"]
            original_prompt = result["experiment_metadata"]["prompt_data"]["prompt"]

            if not fact_mappings:
                continue

            # Extract all sentences from matched_docs once
            all_sentences = []
            for doc in matched_docs:
                if isinstance(doc, dict):
                    doc_content = doc.get('content', str(doc))
                elif isinstance(doc, list):
                    doc_content = '\n'.join([str(item) for item in doc])
                else:
                    doc_content = str(doc)
                sentences = [s.strip() for s in doc_content.split('.') if len(s.strip()) > 20]
                all_sentences.extend(sentences)

            # Prepare arguments for parallel execution
            args_list = []
            for fact_mapping in fact_mappings:
                claim = fact_mapping.get('fact', '')
                if not claim:
                    continue
                args_list.append((claim, all_sentences, original_prompt))

            # Parallel coverage evaluation
            with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
                coverage_evaluations = list(executor.map(evaluate_claim_coverage, args_list))

            # Collect results
            for j, coverage_eval in enumerate(coverage_evaluations):
                claim = fact_mappings[j].get('fact', '')
                coverage_results.append({
                    "experiment_id": result["experiment_metadata"]["experiment_id"],
                    "claim": claim,
                    "covered": coverage_eval["covered"],
                    "supporting_sentences_count": coverage_eval["supporting_sentences_count"]
                })
        # # ...existing code...
        # for i, result in tqdm(enumerate(self.results), total=len(self.results)):
        #     if i % 5 == 0:
        #         print(f"Processing result {i+1}/{len(self.results)}")
            
        #     fact_mappings = result["raw_data"]["fact_mappings"]
        #     matched_docs = result["raw_data"]["matched_docs"]
        #     original_prompt = result["experiment_metadata"]["prompt_data"]["prompt"]
            
        #     if not fact_mappings:
        #         continue
            
        #     # Prepare arguments for parallel execution
        #     args_list = []
        #     for fact_mapping in fact_mappings:
        #         claim = fact_mapping.get('fact', '')
        #         if not claim:
        #             continue
        #         args_list.append((claim, matched_docs, original_prompt))
            
        #     # Parallel coverage evaluation
        #     with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        #         coverage_evaluations = list(executor.map(evaluate_claim_coverage, args_list))
            
        #     # Collect results
        #     for j, coverage_eval in enumerate(coverage_evaluations):
        #         claim = fact_mappings[j].get('fact', '')
                
        #         coverage_results.append({
        #             "experiment_id": result["experiment_metadata"]["experiment_id"],
        #             "claim": claim,
        #             "covered": coverage_eval["covered"],
        #             "supporting_sentences_count": coverage_eval["supporting_sentences_count"]
        #         })
        
        # Calculate aggregate metrics
        total_claims = len(coverage_results)
        covered_claims = sum(1 for r in coverage_results if r["covered"])
        
        metrics = {
            "overall_coverage_rate": covered_claims / total_claims if total_claims > 0 else 0,
            "total_claims_evaluated": total_claims,
            "covered_claims": covered_claims,
            "uncovered_claims": total_claims - covered_claims,
            "avg_supporting_sentences": np.mean([r["supporting_sentences_count"] for r in coverage_results]) if coverage_results else 0
        }
        
        # Plot coverage analysis
        self._plot_coverage_analysis(coverage_results, metrics)
        
        # Profiling
        coverage_time = time.time() - t0
        ic(coverage_time)
        
        return metrics, coverage_results

    def _evaluate_single_claim_coverage(self, claim, sentences, original_prompt):
        """Evaluate coverage for a single claim - check if any sentence supports it"""
        supporting_sentences_count = 0
        for sentence in sentences:
            sentence_eval = LLM_Evaluations_obj._llm_judge_sentence_support(claim, sentence, original_prompt)
            if sentence_eval["supports_claim"]:
                supporting_sentences_count += 1
        covered = supporting_sentences_count > 0
        return {
            "covered": covered,
            "supporting_sentences_count": supporting_sentences_count
        }
    # def _evaluate_single_claim_coverage(self, claim, matched_docs, original_prompt):
    #     """Evaluate coverage for a single claim - check if any sentence supports it"""
    #     supporting_sentences_count = 0
        
    #     # Extract sentences from documents and check if they support the claim
    #     for doc in matched_docs:
    #         if isinstance(doc, dict):
    #             doc_content = doc.get('content', str(doc))
    #         elif isinstance(doc, list):
    #             doc_content = '\n'.join([str(item) for item in doc])
    #         else:
    #             doc_content = str(doc)
            
    #         # Simple sentence splitting
    #         sentences = [s.strip() for s in doc_content.split('.') if len(s.strip()) > 20]
            
    #         # Check each sentence for support (using existing LLM judge)
    #         for sentence in sentences:
    #             sentence_eval = LLM_Evaluations_obj._llm_judge_sentence_support(claim, sentence, original_prompt)
    #             if sentence_eval["supports_claim"]:
    #                 supporting_sentences_count += 1
        
    #     # Claim is covered if at least one supporting sentence is found
    #     covered = supporting_sentences_count > 0
        
    #     return {
    #         "covered": covered,
    #         "supporting_sentences_count": supporting_sentences_count
    #     }
        

    def _plot_coverage_analysis(self, coverage_results, metrics):
        """Plot coverage analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Overall coverage rate
        coverage_rate = metrics["overall_coverage_rate"]
        uncoverage_rate = 1 - coverage_rate
        
        axes[0, 0].pie([coverage_rate, uncoverage_rate],
                    labels=['Covered', 'Uncovered'],
                    autopct='%1.1f%%',
                    colors=['lightgreen', 'lightcoral'])
        axes[0, 0].set_title(f'Overall Coverage Rate\n({metrics["covered_claims"]}/{metrics["total_claims_evaluated"]} claims)')
        
        # Coverage vs uncoverage counts
        axes[0, 1].bar(['Covered', 'Uncovered'], [metrics["covered_claims"], metrics["uncovered_claims"]],
                    color=['lightgreen', 'lightcoral'], alpha=0.7)
        axes[0, 1].set_title('Coverage Counts')
        axes[0, 1].set_ylabel('Number of Claims')
        
        # Supporting sentences count distribution
        supporting_counts = [r["supporting_sentences_count"] for r in coverage_results]
        max_count = max(supporting_counts) if supporting_counts else 1
        
        axes[1, 0].hist(supporting_counts, bins=range(0, max_count + 2), alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Supporting Sentences per Claim')
        axes[1, 0].set_xlabel('Number of Supporting Sentences')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(metrics["avg_supporting_sentences"], color='red', linestyle='--',
                        label=f'Mean: {metrics["avg_supporting_sentences"]:.1f}')
        axes[1, 0].legend()
        
        # Coverage rate summary
        summary_text = f"""
        Coverage Summary:
        
        • Total Claims: {metrics["total_claims_evaluated"]}
        • Covered Claims: {metrics["covered_claims"]}
        • Uncovered Claims: {metrics["uncovered_claims"]}
        • Coverage Rate: {coverage_rate:.1%}
        • Avg Supporting Sentences: {metrics["avg_supporting_sentences"]:.1f}
        """
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        axes[1, 1].set_title('Coverage Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'coverage_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

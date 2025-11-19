import os, json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import concurrent.futures
from icecream import ic
from metrics_calculation_utils.llm_evaluations import LLM_Evaluations
LLM_Evaluations_obj = LLM_Evaluations()

class HitAt5Calculator:
    def __init__(self, results, plots_dir):
        self.results = results
        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)

    def evaluate_hit_at_5_retrieval(self):
        """Evaluate Hit@5 (Top-5 Retrieval Success) using LLM-as-judge"""
        t0 = time.time()
        print("Evaluating Hit@5 retrieval success...")
        
        hit_at_5_results = []

        def judge_hit_at_5(args):
            claim, top_5_docs, original_prompt = args

            relevant_found = False
            doc_relevances = []

            # Judge each of the top-5 docs for relevance to the claim using LLM
            for i, doc in enumerate(top_5_docs[:5]):  # Ensure only top-5
                if isinstance(doc, dict):
                    doc_content = doc.get('content', str(doc))
                elif isinstance(doc, list):
                    doc_content = '\n'.join([str(item) for item in doc])
                else:
                    doc_content = str(doc)

                relevance_result = LLM_Evaluations_obj._llm_judge_sentence_support(claim, doc_content, original_prompt)
                doc_relevances.append({
                    "rank": i + 1,
                    "relevant": relevance_result["supports_claim"],
                    "relevance_score": relevance_result["support_strength"]
                })

                if relevance_result["supports_claim"]:
                    relevant_found = True

            return {
                "hit_at_5": relevant_found,
                "doc_relevances": doc_relevances,
                "first_relevant_rank": next((d["rank"] for d in doc_relevances if d["relevant"]), None)
            }


        for i, result in tqdm(enumerate(self.results), total=len(self.results)):
            if i % 5 == 0:
                print(f"Processing result {i+1}/{len(self.results)}")

            hallucination_results = result["raw_data"].get("hallucination_results", [])
            if not hallucination_results:
                continue    
            # Prepare arguments for parallel execution
            args_list = []
            for hr in hallucination_results:
                claim = hr.get('claim', '')
                topk_sents = hr.get('topk_sents', [])
                topk_sents = [item[0] for item in topk_sents]  # Extract sentences from similarity tuples

                original_prompt = result["experiment_metadata"]["prompt_data"].get("prompt", "")

                if not claim or not topk_sents:
                    continue
                args_list.append((claim, topk_sents, original_prompt))

            # Parallel Hit@5 evaluation
            with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
                hit_evaluations = list(executor.map(judge_hit_at_5, args_list))

            hit_success = []
            hit_miss = []
            # Collect results
            for j, hit_eval in enumerate(hit_evaluations):
                claim = hallucination_results[j].get('claim', '')
                top_5_docs = args_list[j][1]
                original_prompt = args_list[j][2]
                entry = {
                    "experiment_id": result["experiment_metadata"]["experiment_id"],
                    "query": original_prompt,
                    "claim": claim,
                    "top_5_docs": top_5_docs,
                    "hit_at_5": hit_eval["hit_at_5"],
                    "first_relevant_rank": hit_eval["first_relevant_rank"],
                    "doc_relevances": hit_eval["doc_relevances"]
                }
                hit_at_5_results.append(entry)
                if hit_eval["hit_at_5"]:
                    hit_success.append(entry)
                else:
                    hit_miss.append(entry)
            
            # Save results to experimental_results folder
            with open(os.path.join(self.plots_dir, "hit_success.json"), "w") as f:
                json.dump(hit_success, f, indent=2)
            with open(os.path.join(self.plots_dir, "hit_miss.json"), "w") as f:
                json.dump(hit_miss, f, indent=2)


        # Calculate aggregate metrics
        total_queries = len(hit_at_5_results)
        hit_count = sum(1 for r in hit_at_5_results if r["hit_at_5"])
        
        metrics = {
            "overall_hit_at_5_rate": hit_count / total_queries if total_queries > 0 else 0,
            "total_queries_evaluated": total_queries,
            "successful_hits": hit_count,
            "avg_first_relevant_rank": np.mean([r["first_relevant_rank"] for r in hit_at_5_results if r["first_relevant_rank"]]) if any(r["first_relevant_rank"] for r in hit_at_5_results) else None
        }
        
        # Plot Hit@5 analysis
        self._plot_hit_at_5_analysis(hit_at_5_results, metrics)
        
        # Profiling
        hit_at_5_time = time.time() - t0
        ic(hit_at_5_time)
        
        return metrics, hit_at_5_results

    def _plot_hit_at_5_analysis(self, hit_results, metrics):
        """Plot Hit@5 analysis"""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Set larger font sizes globally for this figure
        plt.rcParams.update({'font.size': 14})
        
        # Overall Hit@5 rate (pie chart)
        hit_rate = metrics["overall_hit_at_5_rate"]
        miss_rate = 1 - hit_rate
        
        axes[0].pie([hit_rate, miss_rate], 
                    labels=['Hit@5', 'Miss'], 
                    autopct='%1.1f%%',
                    colors=['lightgreen', 'lightcoral'],
                    textprops={'fontsize': 16})
        axes[0].set_title(f'Overall Hit@5 Success Rate', 
                        fontsize=18, fontweight='bold')
        
        # First relevant rank distribution (for successful hits)
        successful_hits = [r for r in hit_results if r["hit_at_5"] and r["first_relevant_rank"]]
        if successful_hits:
            first_ranks = [r["first_relevant_rank"] for r in successful_hits]
            axes[1].hist(first_ranks, bins=range(1, 7), alpha=0.7, edgecolor='black', color='skyblue')
            axes[1].set_title('First Relevant Document Rank\n(For Successful Hits)', 
                            fontsize=18, fontweight='bold')
            axes[1].set_xlabel('Rank Position', fontsize=16)
            axes[1].set_ylabel('Frequency', fontsize=16)
            axes[1].set_xticks(range(1, 6))
            axes[1].tick_params(axis='both', which='major', labelsize=14)
            
            if metrics["avg_first_relevant_rank"]:
                axes[1].axvline(metrics["avg_first_relevant_rank"], color='red', linestyle='--',
                                linewidth=2, label=f'Mean: {metrics["avg_first_relevant_rank"]:.1f}')
                axes[1].legend(fontsize=14)
        
        # Rank distribution for all relevant documents found
        all_relevant_ranks = []
        for result in hit_results:
            for doc_rel in result["doc_relevances"]:
                if doc_rel["relevant"]:
                    all_relevant_ranks.append(doc_rel["rank"])
        
        if all_relevant_ranks:
            axes[2].hist(all_relevant_ranks, bins=range(1, 7), alpha=0.7, edgecolor='black', color='orange')
            axes[2].set_title('Rank Distribution of All Relevant Documents', 
                            fontsize=18, fontweight='bold')
            axes[2].set_xlabel('Rank Position', fontsize=16)
            axes[2].set_ylabel('Frequency', fontsize=16)
            axes[2].set_xticks(range(1, 6))
            axes[2].tick_params(axis='both', which='major', labelsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'hit_at_5_analysis_3subplots.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.plots_dir, 'hit_at_5_analysis_3subplots.pdf'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Reset font size to default
        plt.rcParams.update({'font.size': plt.rcParamsDefault['font.size']})
    # def _plot_hit_at_5_analysis(self, hit_results, metrics):
    #     """Plot Hit@5 analysis"""
    #     fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
    #     # Overall Hit@5 rate
    #     hit_rate = metrics["overall_hit_at_5_rate"]
    #     miss_rate = 1 - hit_rate
        
    #     axes[0, 0].pie([hit_rate, miss_rate], 
    #                 labels=['Hit@5', 'Miss'], 
    #                 autopct='%1.1f%%',
    #                 colors=['lightgreen', 'lightcoral'])
    #     axes[0, 0].set_title(f'Overall Hit@5 Success Rate\n({metrics["successful_hits"]}/{metrics["total_queries_evaluated"]} queries)')
        
    #     # First relevant rank distribution (for successful hits)
    #     successful_hits = [r for r in hit_results if r["hit_at_5"] and r["first_relevant_rank"]]
    #     if successful_hits:
    #         first_ranks = [r["first_relevant_rank"] for r in successful_hits]
    #         axes[0, 1].hist(first_ranks, bins=range(1, 7), alpha=0.7, edgecolor='black')
    #         axes[0, 1].set_title('First Relevant Document Rank\n(For Successful Hits)')
    #         axes[0, 1].set_xlabel('Rank Position')
    #         axes[0, 1].set_ylabel('Frequency')
    #         axes[0, 1].set_xticks(range(1, 6))
            
    #         if metrics["avg_first_relevant_rank"]:
    #             axes[0, 1].axvline(metrics["avg_first_relevant_rank"], color='red', linestyle='--',
    #                             label=f'Mean: {metrics["avg_first_relevant_rank"]:.1f}')
    #             axes[0, 1].legend()
        
    #     # Hit@5 success vs failure counts
    #     axes[1, 0].bar(['Success', 'Miss'], [metrics["successful_hits"], metrics["total_queries_evaluated"] - metrics["successful_hits"]],
    #                 color=['lightgreen', 'lightcoral'], alpha=0.7)
    #     axes[1, 0].set_title('Hit@5 Success vs Miss Counts')
    #     axes[1, 0].set_ylabel('Number of Queries')
        
    #     # Rank distribution for all relevant documents found
    #     all_relevant_ranks = []
    #     for result in hit_results:
    #         for doc_rel in result["doc_relevances"]:
    #             if doc_rel["relevant"]:
    #                 all_relevant_ranks.append(doc_rel["rank"])
        
    #     if all_relevant_ranks:
    #         axes[1, 1].hist(all_relevant_ranks, bins=range(1, 7), alpha=0.7, edgecolor='black')
    #         axes[1, 1].set_title('Rank Distribution of All Relevant Documents')
    #         axes[1, 1].set_xlabel('Rank Position')
    #         axes[1, 1].set_ylabel('Frequency')
    #         axes[1, 1].set_xticks(range(1, 6))
        
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(self.plots_dir, 'hit_at_5_analysis.png'), dpi=300, bbox_inches='tight')
    #     plt.close()

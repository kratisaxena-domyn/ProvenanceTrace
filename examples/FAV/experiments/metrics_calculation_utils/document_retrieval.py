import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import concurrent.futures
from icecream import ic
from metrics_calculation_utils.llm_evaluations import LLM_Evaluations

LLM_Evaluations_obj = LLM_Evaluations()
class DocumentRetrievalCalculator:
    def __init__(self, results, plots_dir):
        self.results = results
        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)

    def evaluate_document_retrieval(self):
        """Evaluate document retrieval quality using LLM-as-judge"""
        t0 = time.time()
        print("Evaluating document retrieval quality...")
        
        retrieval_results = []
        ic(len(self.results))
        
        def judge_doc(args):
            claim, doc, original_prompt = args
            if isinstance(doc, dict):
                doc_content = doc.get('content', str(doc))
            elif isinstance(doc, list):
                doc_content = '\n'.join([str(item) for item in doc])
            else:
                doc_content = str(doc)
            return LLM_Evaluations_obj._llm_judge_document_relevance(claim, doc_content, original_prompt)
        
        for i, result in tqdm(enumerate(self.results), total=len(self.results)):
            fact_mappings = result["raw_data"].get("fact_mappings", [])
            halluc_results = result["raw_data"].get("hallucination_results", {})
            trace_steps = result["raw_data"].get("trace_steps", [])
            original_prompt = result["experiment_metadata"]["prompt_data"].get("prompt", "")

            # Fix: handle both dict and list types for halluc_results
            if isinstance(halluc_results, dict):
                topk_sents_all = halluc_results.get("topk_sents", [])
            elif isinstance(halluc_results, list):
                topk_sents_all = halluc_results  # assume the list itself is topk_sents
            else:
                topk_sents_all = []
            for fact_idx, fact_mapping in enumerate(fact_mappings):
                claim = fact_mapping.get('fact', '')
                if not claim:
                    continue

                # Get top-5 sentences for this claim
                retrieved_sents = []
                if len(topk_sents_all) > fact_idx:
                    # Flatten and filter out empty lists
                    retrieved_sents = [s for s in topk_sents_all[fact_idx] if s]

                # Judge relevance for each retrieved sentence
                args_list = [(claim, sent, original_prompt) for sent in retrieved_sents]
                with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
                    doc_evals = list(executor.map(judge_doc, args_list))

                doc_evaluations = [
                    {
                        "rank": j + 1,
                        "relevant": eval_result["relevant"],
                        "relevance_score": eval_result["relevance_score"],
                        "reasoning": eval_result["reasoning"]
                    }
                    for j, eval_result in enumerate(doc_evals)
                ]
                relevant_docs = [d for d in doc_evaluations if d["relevant"]]

                # Get all candidate sentences for recall denominator
                candidate_docs = []
                if isinstance(trace_steps, list) and len(trace_steps) > fact_idx:
                    raw_doc = trace_steps[fact_idx]
                    if raw_doc is None:
                        candidate_docs = []
                    elif isinstance(raw_doc, str):
                        candidate_docs = [s.strip() for s in raw_doc.split('\n') if s.strip()]
                    elif isinstance(raw_doc, dict):
                        candidate_docs = [' '.join([f"{k}: {v}" for k, v in raw_doc.items()])]
                    elif isinstance(raw_doc, list):
                        for item in raw_doc:
                            if isinstance(item, str):
                                candidate_docs.append(item.strip())
                            elif isinstance(item, dict):
                                candidate_docs.append(' '.join([f"{k}: {v}" for k, v in item.items()]))
                            else:
                                candidate_docs.append(str(item))
                    else:
                        candidate_docs = [str(raw_doc)]
                else:
                    candidate_docs = []

                # Judge relevance for each candidate doc
                args_list_gt = [(claim, doc, original_prompt) for doc in candidate_docs]
                with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
                    gt_evals = list(executor.map(judge_doc, args_list_gt))
                num_ground_truth = sum([eval_result["relevant"] for eval_result in gt_evals])

                precision_at_5 = len(relevant_docs) / 5 # if retrieved_sents else 0
                recall_at_5 = len(relevant_docs) / num_ground_truth if num_ground_truth > 0 else 0

                retrieval_results.append({
                    "experiment_id": result["experiment_metadata"].get("experiment_id", None),
                    "claim": claim,
                    "total_docs_retrieved": len(doc_evaluations),
                    "relevant_docs_found": len(relevant_docs),
                    "precision_at_5": precision_at_5,
                    "recall_at_5": recall_at_5,
                    "mrr": (1.0 / relevant_docs[0]["rank"]) if relevant_docs else 0,
                    "doc_evaluations": doc_evaluations
                })


        # Calculate aggregate metrics
        metrics = {
            "avg_precision_at_5": np.mean([r["precision_at_5"] for r in retrieval_results]),
            "avg_recall_at_5": np.mean([r["recall_at_5"] for r in retrieval_results]),
            "avg_mrr": np.mean([r["mrr"] for r in retrieval_results]),
            "total_evaluations": len(retrieval_results)
        }
        
        # Plot document retrieval analysis
        self._plot_retrieval_analysis(retrieval_results, metrics)
        
        # Profiling
        llm_doc_retrieval_time = time.time() - t0
        ic(llm_doc_retrieval_time)
        
        return metrics, retrieval_results

    def _plot_retrieval_analysis(self, retrieval_results, metrics):
        """Plot document retrieval analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Precision@5 distribution
        precisions = [r["precision_at_5"] for r in retrieval_results]
        axes[0, 0].hist(precisions, bins=15, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Precision@5 Distribution')
        axes[0, 0].set_xlabel('Precision@5')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(metrics["avg_precision_at_5"], color='red', linestyle='--',
                          label=f'Mean: {metrics["avg_precision_at_5"]:.3f}')
        axes[0, 0].legend()
        
        # MRR distribution
        mrrs = [r["mrr"] for r in retrieval_results]
        axes[0, 1].hist(mrrs, bins=15, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Mean Reciprocal Rank Distribution')
        axes[0, 1].set_xlabel('MRR')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(metrics["avg_mrr"], color='red', linestyle='--',
                          label=f'Mean: {metrics["avg_mrr"]:.3f}')
        axes[0, 1].legend()
        
        # Relevant docs found distribution
        relevant_counts = [r["relevant_docs_found"] for r in retrieval_results]
        axes[1, 0].hist(relevant_counts, bins=range(0, max(relevant_counts) + 2), 
                       alpha=0.7, edgecolor='black')
        # axes[1, 0].set_title('Relevant Documents Found (per claim)')
        axes[1, 0].set_title(f'Relevant Documents Found (out of {max([len(r["doc_evaluations"]) for r in retrieval_results])})')
        axes[1, 0].set_xlabel('Number of Relevant Documents')
        axes[1, 0].set_ylabel('Frequency')
        
        # Summary metrics
        summary_metrics = ['Precision@5', 'Recall@5', 'MRR']
        summary_values = [metrics["avg_precision_at_5"], metrics["avg_recall_at_5"], metrics["avg_mrr"]]
        
        bars = axes[1, 1].bar(summary_metrics, summary_values, alpha=0.7, color=['skyblue', 'lightgreen', 'orange'])
        axes[1, 1].set_title('Document Retrieval Metrics Summary')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, summary_values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'document_retrieval_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

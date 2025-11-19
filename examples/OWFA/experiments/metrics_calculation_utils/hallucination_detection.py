import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import time
import concurrent.futures
from icecream import ic
from metrics_calculation_utils.llm_evaluations import LLM_Evaluations
import pandas as pd
LLM_Evaluations_obj = LLM_Evaluations()

class HallucinationDetectionCalculator:
    def __init__(self, results, plots_dir):
        self.results = results
        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)

    def evaluate_hallucination_detection(self):
        """Evaluate hallucination detection using hallucination_results JSON"""
        t0 = time.time()
        print("Evaluating hallucination detection...")

        # Collect all hallucination_results from each experiment
        all_halluc_results = []
        for result in tqdm(self.results, total=len(self.results)):
            halluc_results = result["raw_data"].get("hallucination_results", [])
            for hr in halluc_results:
                hr["experiment_id"] = result["experiment_metadata"]["experiment_id"]
                all_halluc_results.append(hr)

        # Prepare metrics for analysis
        max_sims = [r["max_sim"] for r in all_halluc_results if "max_sim" in r]
        avg_topk_sims = [r["avg_topk_sim"] for r in all_halluc_results if "avg_topk_sim" in r]
        entailment_scores = [r.get("entailment_score", None) for r in all_halluc_results if "entailment_score" in r]
        statuses = [r.get("status", "Unknown") for r in all_halluc_results]
        topk_sim_flat = [sim for r in all_halluc_results for sims in r.get("topk_sents", []) for sim in sims if isinstance(sim, float)]

        # Plot analysis
        self._plot_hallucination_json_analysis(max_sims, avg_topk_sims, entailment_scores, statuses, topk_sim_flat)

        # Profiling
        hallucination_detection_time = time.time() - t0
        ic(hallucination_detection_time)

        # Aggregate metrics
        metrics = {
            "total_claims": len(all_halluc_results),
            "mean_max_sim": np.mean(max_sims) if max_sims else 0,
            "mean_avg_topk_sim": np.mean(avg_topk_sims) if avg_topk_sims else 0,
            "mean_entailment_score": np.mean([e for e in entailment_scores if e is not None]) if entailment_scores else 0,
            "status_counts": dict(zip(*np.unique(statuses, return_counts=True)))
        }
        return metrics, all_halluc_results

    def _plot_hallucination_json_analysis(self, max_sims, avg_topk_sims, entailment_scores, statuses, topk_sim_flat):
        """Plot analysis for hallucination_results JSON"""
        # Set font sizes
        plt.rcParams.update({'font.size': 14})
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # 1. Combined histogram of max_sim and avg_topk_sim
        axes[0].hist(max_sims, bins=20, alpha=0.7, label='max_sim', color='skyblue', edgecolor='black')
        axes[0].hist(avg_topk_sims, bins=20, alpha=0.7, label='avg_topk_sim', color='orange', edgecolor='black')
        axes[0].set_title('Distribution of Similarity Scores', fontsize=16, fontweight='bold')
        axes[0].set_xlabel('Similarity Score', fontsize=14)
        axes[0].set_ylabel('Frequency', fontsize=14)
        axes[0].legend(fontsize=12)
        axes[0].tick_params(labelsize=12)

        # Option 4: Binned scatter plot with error bars
                # Option 4: Binned scatter plot with error bars
        valid_entailment = [(max_sim, ent_score) for max_sim, ent_score in zip(max_sims, entailment_scores) if ent_score is not None]
        if valid_entailment:
            max_sims_valid, entailment_scores_valid = zip(*valid_entailment)
            # Create bins for max_sim
            bins = np.linspace(min(max_sims_valid), max(max_sims_valid), 10)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            bin_means = []
            bin_stds = []
            
            for i in range(len(bins)-1):
                mask = (np.array(max_sims_valid) >= bins[i]) & (np.array(max_sims_valid) < bins[i+1])
                if np.any(mask):
                    bin_entailments = np.array(entailment_scores_valid)[mask]
                    bin_means.append(np.mean(bin_entailments))
                    bin_stds.append(np.std(bin_entailments))
                else:
                    bin_means.append(0)
                    bin_stds.append(0)
            
            # Clamp error bars so they don't go below 0.0
            bin_means = np.array(bin_means)
            bin_stds = np.array(bin_stds)
            lower_errors = np.minimum(bin_stds, bin_means)  # Don't let error bars go below 0
            
            axes[1].errorbar(bin_centers, bin_means, yerr=[lower_errors, bin_stds], fmt='o-', 
                            color='purple', capsize=5, capthick=2, linewidth=2, markersize=8)
            axes[1].set_title('Entailment Score vs Max Similarity (Binned)', fontsize=16, fontweight='bold')
            axes[1].set_xlabel('max_sim', fontsize=14)
            axes[1].set_ylabel('Mean entailment_score', fontsize=14)
            
        # 3. Bar chart of hallucination status counts
        unique_statuses, status_counts = np.unique(statuses, return_counts=True)
        bars = axes[2].bar(unique_statuses, status_counts, color='mediumseagreen', alpha=0.8, edgecolor='black')
        axes[2].set_title('Hallucination Status Distribution', fontsize=16, fontweight='bold')
        axes[2].set_xlabel('Status', fontsize=14)
        axes[2].set_ylabel('Count', fontsize=14)
        axes[2].tick_params(labelsize=12)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            axes[2].annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=11)

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'hallucination_json_analysis_3subplots.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.plots_dir, 'hallucination_json_analysis_3subplots.pdf'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Reset font size to default
        plt.rcParams.update({'font.size': plt.rcParamsDefault['font.size']})
        
    # def _plot_hallucination_json_analysis(self, max_sims, avg_topk_sims, entailment_scores, statuses, topk_sim_flat):
    #     """Plot analysis for hallucination_results JSON"""
    #     fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    #     # 1. Histogram of max_sim
    #     axes[0, 0].hist(max_sims, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    #     axes[0, 0].set_title('Distribution of max_sim')
    #     axes[0, 0].set_xlabel('max_sim')
    #     axes[0, 0].set_ylabel('Frequency')

    #     # 2. Histogram of avg_topk_sim
    #     axes[0, 1].hist(avg_topk_sims, bins=20, color='orange', edgecolor='black', alpha=0.7)
    #     axes[0, 1].set_title('Distribution of avg_topk_sim')
    #     axes[0, 1].set_xlabel('avg_topk_sim')
    #     axes[0, 1].set_ylabel('Frequency')

    #     # 3. Scatter plot: entailment_score vs max_sim
    #     axes[1, 0].scatter(max_sims, entailment_scores, alpha=0.6, color='purple')
    #     axes[1, 0].set_title('Entailment Score vs max_sim')
    #     axes[1, 0].set_xlabel('max_sim')
    #     axes[1, 0].set_ylabel('entailment_score')

    #     # 4. Bar chart of hallucination status counts
    #     unique_statuses, status_counts = np.unique(statuses, return_counts=True)
    #     axes[1, 1].bar(unique_statuses, status_counts, color='green', alpha=0.7)
    #     axes[1, 1].set_title('Hallucination Status Counts')
    #     axes[1, 1].set_xlabel('Status')
    #     axes[1, 1].set_ylabel('Count')

    #     plt.tight_layout()
    #     plt.savefig(os.path.join(self.plots_dir, 'hallucination_json_analysis.png'), dpi=300, bbox_inches='tight')
    #     plt.close()

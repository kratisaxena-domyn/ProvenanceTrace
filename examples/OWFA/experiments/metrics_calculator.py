import os, sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from icecream import ic
import concurrent.futures
import argparse
from collections import defaultdict
from openai import OpenAI
import glob
import time
from tqdm import tqdm

from metrics_calculation_utils.coverage import CoverageCalculator
from metrics_calculation_utils.document_retrieval import DocumentRetrievalCalculator
from metrics_calculation_utils.hallucination_detection import HallucinationDetectionCalculator
from metrics_calculation_utils.hit_at_5 import HitAt5Calculator
from metrics_calculation_utils.sentence_matching import SentenceMatchingCalculator
from metrics_calculation_utils.empirical_general_analysis import GeneralAnalysis
from .utils.config_loader import get_llm_client_config

llm_config = get_llm_client_config()

class EnhancedMetricsCalculator:
    def __init__(self, results_data_or_path):
        """Initialize with either results data or path to experiment results directory"""
        if isinstance(results_data_or_path, str):
            # Load results from directory containing prompt batch files
            self.results = self._load_results_from_directory(results_data_or_path)
        else:
            # Direct results data
            self.results = results_data_or_path
        
        # Create plots directory
        self.plots_dir = os.path.join(results_data_or_path, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Initialize LLM client for evaluation
        self.client = OpenAI(
            base_url=llm_config["base_url"],
            api_key=llm_config["api_key"],
        )
        self.evaluate_coverage_rate_obj = CoverageCalculator(self.results, self.plots_dir)
        self.evaluate_document_retrieval_obj = DocumentRetrievalCalculator(self.results, self.plots_dir)    
        self.hallucination_detection_obj = HallucinationDetectionCalculator(self.results, self.plots_dir)
        self.hit_at_5_calculator_obj = HitAt5Calculator(self.results, self.plots_dir)   
        self.sentence_matching_calculator_obj = SentenceMatchingCalculator(self.results, self.plots_dir)    
        self.general_analysis_obj = GeneralAnalysis(self.results, self.plots_dir)   
        
    def run_document_retrieval(self):
        print("Running document retrieval evaluation...")
        return self.evaluate_document_retrieval_obj.evaluate_document_retrieval()

    def run_sentence_matching(self):
        print("Running sentence matching evaluation...")
        return self.sentence_matching_calculator_obj.evaluate_sentence_matching()

    def run_hallucination_detection(self):
        print("Running hallucination detection evaluation...")
        return self.hallucination_detection_obj.evaluate_hallucination_detection()

    def run_hit_at_5(self):
        print("Running Hit@5 evaluation...")
        return self.hit_at_5_calculator_obj.evaluate_hit_at_5_retrieval()

    def run_coverage_rate(self):
        print("Running coverage rate evaluation...")
        return self.evaluate_coverage_rate_obj.evaluate_coverage_rate()

    # def run_general_analysis(self):
    #     print("Running general analysis...")
    #     return self.general_analysis_obj.calculate_system_performance_metrics()
    def run_general_analysis(self):
        print("Running general analysis...")
        print("Calculating system performance metrics...")
        system_metrics = self.general_analysis_obj.calculate_system_performance_metrics()
        
        print("Calculating claim verification metrics...")
        claim_metrics, confidence_df = self.general_analysis_obj.calculate_claim_verification_metrics()
        
        # print("Calculating provenance metrics...")
        # provenance_metrics, trace_df = self.general_analysis_obj.calculate_provenance_metrics()
        
        print("Calculating stability metrics...")
        stability_metrics, stability_df = self.general_analysis_obj.calculate_stability_metrics()
        
        # Combine all metrics into a comprehensive report
        general_analysis_report = {
            "system_performance": system_metrics,
            "claim_verification": claim_metrics,
            # "provenance_quality": provenance_metrics,
            "output_stability": stability_metrics,
            "summary": {
                "total_experiments": len(self.results),
                "overall_success_rate": system_metrics["success_rates"].get("trace_success", 0),
                "claim_support_rate": claim_metrics["support_rate"],
                "avg_response_time": system_metrics["runtime_stats"]["mean"],
                # "trace_completeness_rate": provenance_metrics["trace_completeness_rate"],
                "stable_prompts_ratio": stability_metrics["stable_prompts_ratio"]
            }
        }
        
        print("\n" + "="*60)
        print("GENERAL ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total experiments: {len(self.results)}")
        print(f"Average response time: {system_metrics['runtime_stats']['mean']:.2f}s")
        print(f"Claim support rate: {claim_metrics['support_rate']:.2%}")
        # print(f"Trace completeness rate: {provenance_metrics['trace_completeness_rate']:.2%}")
        print(f"Stable prompts ratio: {stability_metrics['stable_prompts_ratio']:.2%}")
        print("="*60)
        
        return general_analysis_report, {
            "confidence_df": confidence_df,
            # "trace_df": trace_df,
            "stability_df": stability_df
        }
    
    
    def _load_results_from_directory(self, results_dir):
        """Load all experiment results from prompt batch files"""
        t0 = time.time()
        results = []
        batch_files = glob.glob(os.path.join(results_dir, "*_prompt_*_all_runs.json"))
        for batch_file in tqdm(batch_files):
            with open(batch_file, "r") as f:
                batch_data = json.load(f)
                results.extend(batch_data)
        load_res_time = time.time() - t0
        ic(load_res_time)
        return results

    def _plot_enhanced_summary_dashboard(self, retrieval_metrics, sentence_metrics, halluc_metrics, 
                                   hit_at_5_metrics, coverage_metrics):
        """Create enhanced summary dashboard with all LLM-based metrics"""
        t0 = time.time()
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        # Document retrieval metrics
        retrieval_names = ['Precision@5', 'Recall@5', 'MRR']
        retrieval_values = [
            retrieval_metrics.get("avg_precision_at_5", 0),
            retrieval_metrics.get("avg_recall_at_5", 0),
            retrieval_metrics.get("avg_mrr", 0)
        ]
        
        bars1 = axes[0, 0].bar(retrieval_names, retrieval_values, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Document Retrieval Quality')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_ylim(0, 1)
        
        for bar, value in zip(bars1, retrieval_values):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # Hit@5 metrics
        axes[0, 1].pie([hit_at_5_metrics.get("overall_hit_at_5_rate", 0), 
                    1 - hit_at_5_metrics.get("overall_hit_at_5_rate", 0)],
                    labels=['Hit@5', 'Miss'], autopct='%1.1f%%',
                    colors=['lightgreen', 'lightcoral'])
        axes[0, 1].set_title(f'Hit@5 Success Rate\n{hit_at_5_metrics.get("successful_hits", 0)}/{hit_at_5_metrics.get("total_queries_evaluated", 0)} queries')
        
        # Coverage rate
        axes[0, 2].pie([coverage_metrics.get("overall_coverage_rate", 0),
                    1 - coverage_metrics.get("overall_coverage_rate", 0)],
                    labels=['Covered', 'Uncovered'], autopct='%1.1f%%',
                    colors=['lightgreen', 'lightcoral'])
        axes[0, 2].set_title(f'Coverage Rate\n{coverage_metrics.get("covered_claims", 0)}/{coverage_metrics.get("total_claims_evaluated", 0)} claims')
        
        # Hit@5 success breakdown
        axes[1, 0].bar(['Success', 'Miss'], 
                    [hit_at_5_metrics.get("successful_hits", 0), 
                    hit_at_5_metrics.get("total_queries_evaluated", 0) - hit_at_5_metrics.get("successful_hits", 0)],
                    alpha=0.7, color=['lightgreen', 'lightcoral'])
        axes[1, 0].set_title('Hit@5 Success Breakdown')
        axes[1, 0].set_ylabel('Number of Queries')
        
        # Coverage breakdown
        axes[1, 1].bar(['Covered', 'Uncovered'], 
                    [coverage_metrics.get("covered_claims", 0), coverage_metrics.get("uncovered_claims", 0)],
                    alpha=0.7, color=['lightgreen', 'lightcoral'])
        axes[1, 1].set_title('Coverage Breakdown')
        axes[1, 1].set_ylabel('Number of Claims')
        
        # Sentence matching metrics
        sentence_names = ['Precision', 'Support Strength']
        sentence_values = [
            sentence_metrics.get("avg_sentence_precision", 0),
            sentence_metrics.get("avg_support_strength", 0)
        ]
        
        bars4 = axes[1, 2].bar(sentence_names, sentence_values, alpha=0.7, color='lightgreen')
        axes[1, 2].set_title('Sentence-Level Matching')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].set_ylim(0, 1)
        
        for bar, value in zip(bars4, sentence_values):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # Hallucination risk distribution
        risk_dist = halluc_metrics.get("risk_distribution", {})
        if risk_dist:
            axes[2, 0].pie(risk_dist.values(), labels=risk_dist.keys(), autopct='%1.1f%%',
                        colors=['lightgreen', 'orange', 'lightcoral'])
            axes[2, 0].set_title('Hallucination Risk Distribution')
        
        # Overall quality scores comparison
        quality_metrics = {
            'Hit@5': hit_at_5_metrics.get("overall_hit_at_5_rate", 0),
            'Coverage': coverage_metrics.get("overall_coverage_rate", 0),
            'Precision@5': retrieval_metrics.get("avg_precision_at_5", 0),
            'Sentence': sentence_metrics.get("avg_sentence_precision", 0)
        }
        
        bars5 = axes[2, 1].bar(quality_metrics.keys(), quality_metrics.values(), 
                            alpha=0.7, color=['gold', 'lightblue', 'lightgreen', 'orange'])
        axes[2, 1].set_title('Key Quality Metrics Comparison')
        axes[2, 1].set_ylabel('Score')
        axes[2, 1].set_ylim(0, 1)
        axes[2, 1].tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars5, quality_metrics.values()):
            height = bar.get_height()
            axes[2, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # Key statistics summary
        key_stats = f"""
        Enhanced LLM Evaluations Summary:
        
        Hit@5 Retrieval:
        • Overall Success: {hit_at_5_metrics.get("overall_hit_at_5_rate", 0):.1%}
        • Total Queries: {hit_at_5_metrics.get("total_queries_evaluated", 0)}
        • Avg First Relevant Rank: {hit_at_5_metrics.get("avg_first_relevant_rank", 0):.1f}
        
        Coverage Analysis:
        • Overall Coverage: {coverage_metrics.get("overall_coverage_rate", 0):.1%}
        • Total Claims: {coverage_metrics.get("total_claims_evaluated", 0)}
        • Avg Supporting Sentences: {coverage_metrics.get("avg_supporting_sentences", 0):.1f}
        
        Document Retrieval:
        • Precision@5: {retrieval_metrics.get("avg_precision_at_5", 0):.3f}
        • MRR: {retrieval_metrics.get("avg_mrr", 0):.3f}
        
        Quality Assessment:
        • High Risk Rate: {halluc_metrics.get("high_risk_rate", 0):.1%}
        • Sentence Precision: {sentence_metrics.get("avg_sentence_precision", 0):.3f}
        """
        
        axes[2, 2].text(0.05, 0.95, key_stats, transform=axes[2, 2].transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        axes[2, 2].set_title('Enhanced Metrics Summary')
        axes[2, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'enhanced_summary_dashboard.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        generate_enhanced_summary_time = time.time() - t0
        ic(generate_enhanced_summary_time)
    
    
    
    # Add to generate_enhanced_report method:
    def generate_enhanced_report(self):
        """Generate comprehensive report including new LLM-based evaluations"""
        t0 = time.time()
        print("Generating enhanced metrics report with LLM-based evaluations...")
        
        # Original metrics
        print("Calculating basic system metrics...")
        system_metrics = self.general_analysis_obj.calculate_system_performance_metrics()
        claim_metrics, confidence_df = self.general_analysis_obj.calculate_claim_verification_metrics()
        provenance_metrics, trace_df = self.general_analysis_obj.calculate_provenance_metrics()
        stability_metrics, stability_df = self.general_analysis_obj.calculate_stability_metrics()
        
        # Enhanced LLM-based evaluations
        print("Starting LLM-based evaluations...")
        retrieval_metrics, retrieval_data = self.evaluate_document_retrieval_obj.evaluate_document_retrieval()
        sentence_metrics, sentence_data = self.sentence_matching_calculator_obj.evaluate_sentence_matching()
        halluc_metrics, halluc_data = self.hallucination_detection_obj.evaluate_hallucination_detection()
        
        # New metrics
        print("Evaluating Hit@5 and Coverage metrics...")
        hit_at_5_metrics, hit_at_5_data = self.hit_at_5_calculator_obj.evaluate_hit_at_5_retrieval()
        coverage_metrics, coverage_data = self.evaluate_coverage_rate_obj.evaluate_coverage_rate()
        
        # Create enhanced summary dashboard
        self._plot_enhanced_summary_dashboard(retrieval_metrics, sentence_metrics, halluc_metrics, 
                                            hit_at_5_metrics, coverage_metrics)
        
        enhanced_report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_experiments": len(self.results),
                "llm_evaluations_completed": True,
                "plots_saved_to": self.plots_dir
            },
            "llm_based_evaluations": {
                "document_retrieval": retrieval_metrics,
                "sentence_matching": sentence_metrics,
                "hallucination_detection": halluc_metrics,
                "hit_at_5": hit_at_5_metrics,
                "coverage_rate": coverage_metrics
            },
            "summary": {
                "document_precision_at_5": retrieval_metrics.get("avg_precision_at_5", 0),
                "sentence_precision": sentence_metrics.get("avg_sentence_precision", 0),
                "hallucination_high_risk_rate": halluc_metrics.get("high_risk_rate", 0),
                "hit_at_5_rate": hit_at_5_metrics.get("overall_hit_at_5_rate", 0),
                "coverage_rate": coverage_metrics.get("overall_coverage_rate", 0),
                "avg_support_strength": sentence_metrics.get("avg_support_strength", 0)
            }
        }
        
        generate_enhanced_report_time = time.time() - t0
        ic(generate_enhanced_report_time)
        
        return enhanced_report, {
            "retrieval_data": retrieval_data,
            "sentence_data": sentence_data,
            "hallucination_data": halluc_data,
            "hit_at_5_data": hit_at_5_data,
            "coverage_data": coverage_data
        }
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run metrics calculation for experiment results.")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing experiment results")
    parser.add_argument("--experiment_name", type=str, default="experiment", help="Name for the experiment")
    parser.add_argument("--skip_llm_eval", action="store_true", help="Skip LLM-based evaluation")
    parser.add_argument("--run_evaluation", type=str, default=None,
        help="Run a single evaluation: document_retrieval, sentence_matching, hallucination_detection, hit_at_5, coverage_rate, general_analysis")
    args = parser.parse_args()

    calculator = EnhancedMetricsCalculator(args.results_dir)

    if len(calculator.results) == 0:
        print(f"No experiment results found in {args.results_dir}")
        print("Please run comprehensive_experiment_runner.py first to generate results")
        exit(1)

    
    
    ic(args)
    if hasattr(args, 'run_evaluation') and args.run_evaluation:
        eval_map = {
            "document_retrieval": calculator.run_document_retrieval,
            "sentence_matching": calculator.run_sentence_matching,
            "hallucination_detection": calculator.run_hallucination_detection,
            "hit_at_5": calculator.run_hit_at_5,
            "coverage_rate": calculator.run_coverage_rate,
            "general_analysis": calculator.run_general_analysis
        }
        if args.run_evaluation not in eval_map:
            print(f"Unknown evaluation: {args.run_evaluation}")
            print("Valid options: document_retrieval, sentence_matching, hallucination_detection, hit_at_5, coverage_rate, general_analysis")
            sys.exit(1)
        print(f"Running {args.run_evaluation}...")
        result = eval_map[args.run_evaluation]()
    elif args.skip_llm_eval:
        print("Skipping LLM-based evaluations...")
        # Run original metrics only
        system_metrics = calculator.general_analysis_obj.calculate_system_performance_metrics()
        claim_metrics, confidence_df = calculator.general_analysis_obj.calculate_claim_verification_metrics()
        provenance_metrics, trace_df = calculator.general_analysis_obj.calculate_provenance_metrics()
        stability_metrics, stability_df = calculator.general_analysis_obj.calculate_stability_metrics()

        # Generate summary visualization
        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_experiments": len(calculator.results),
                "plots_saved_to": calculator.plots_dir
            },
            "system_performance": system_metrics,
            "claim_verification": claim_metrics,
            "provenance_quality": provenance_metrics,
            "output_stability": stability_metrics,
            "summary": {
                "total_experiments": len(calculator.results),
                "overall_success_rate": system_metrics["success_rates"].get("agent_success", 0),
                "claim_support_rate": claim_metrics["support_rate"],
                "avg_response_time": system_metrics["runtime_stats"]["mean"],
                "trace_completeness_rate": provenance_metrics["trace_completeness_rate"]
            }
        }

        report_file = os.path.join(args.results_dir, f"{args.experiment_name}_metrics_report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print("\n" + "="*60)
        print("EXPERIMENT METRICS SUMMARY")
        print("="*60)
        print(json.dumps(report["summary"], indent=2))
        print(f"\nDetailed report saved to: {report_file}")
        print(f"Plots saved to: {calculator.plots_dir}")
        print("\nGenerated plots:")
        for plot_file in os.listdir(calculator.plots_dir):
            if plot_file.endswith('.png'):
                print(f"  - {plot_file}")
    else:
        # Generate enhanced report with LLM evaluations
        print("Running enhanced evaluation with LLM-as-judge...")
        print(f"This will evaluate {len(calculator.results)} experiments using LLM judgments")
        print("Note: This may take several minutes due to LLM API calls...")

        enhanced_report, enhanced_data = calculator.generate_enhanced_report()

        # Save enhanced report
        report_file = os.path.join(args.results_dir, f"{args.experiment_name}_enhanced_metrics_report.json")
        with open(report_file, "w") as f:
            json.dump(enhanced_report, f, indent=2, default=str)

        print("\n" + "="*60)
        print("ENHANCED EXPERIMENT METRICS SUMMARY")
        print("="*60)
        print(json.dumps(enhanced_report["summary"], indent=2))
        print(f"\nDetailed report saved to: {report_file}")
        print(f"Plots saved to: {calculator.plots_dir}")
        print("\nGenerated enhanced plots:")
        for plot_file in os.listdir(calculator.plots_dir):
            if plot_file.endswith('.png'):
                print(f"  - {plot_file}")
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from icecream import ic

class GeneralAnalysis:
    def __init__(self, results, plots_dir):
        self.results = results
        self.plots_dir = plots_dir

    def _plot_success_rates(self, success_rates):
        """Plot success rates for different components"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Success rates bar chart
        components = list(success_rates.keys())
        rates = [np.mean(success_rates[comp]) for comp in components]
        
        bars = ax1.bar([comp.replace('_success', '').replace('_', ' ').title() for comp in components], 
                      rates, alpha=0.7)
        ax1.set_title('Component Success Rates')
        ax1.set_ylabel('Success Rate')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.2f}', ha='center', va='bottom')
        
        ax1.tick_params(axis='x', rotation=45)
        
        # Success rate trends over experiments
        for component in components:
            successes = success_rates[component]
            # Calculate rolling average
            window_size = min(10, len(successes) // 3)
            if window_size > 1:
                rolling_avg = pd.Series(successes).rolling(window=window_size).mean()
                ax2.plot(range(len(rolling_avg)), rolling_avg, 
                        label=component.replace('_success', '').replace('_', ' ').title(), 
                        alpha=0.7)
        
        ax2.set_title('Success Rate Trends (Rolling Average)')
        ax2.set_xlabel('Experiment Index')
        ax2.set_ylabel('Success Rate')
        ax2.legend()
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'success_rates.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_stability_analysis(self, stability_df):
        """Plot stability analysis across multiple runs"""
        if len(stability_df) == 0:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Coefficient of variation distributions
        cv_metrics = ['answer_length_cv', 'fact_count_cv', 'support_rate_cv']
        cv_labels = ['Answer Length CV', 'Fact Count CV', 'Support Rate CV']
        
        for i, (metric, label) in enumerate(zip(cv_metrics, cv_labels)):
            axes[0, 0].hist(stability_df[metric], bins=15, alpha=0.7, label=label)
        
        axes[0, 0].set_title('Coefficient of Variation Distributions')
        axes[0, 0].set_xlabel('Coefficient of Variation')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # Stability scatter plot
        axes[0, 1].scatter(stability_df['fact_count_cv'], stability_df['support_rate_cv'], alpha=0.6)
        axes[0, 1].set_title('Fact Count Stability vs Support Rate Stability')
        axes[0, 1].set_xlabel('Fact Count CV')
        axes[0, 1].set_ylabel('Support Rate CV')
        
        # Stable vs unstable prompts
        stable_threshold = 0.2
        stable_prompts = (stability_df['support_rate_cv'] < stable_threshold).sum()
        unstable_prompts = len(stability_df) - stable_prompts
        
        axes[1, 0].bar(['Stable', 'Unstable'], [stable_prompts, unstable_prompts],
                      color=['lightgreen', 'lightcoral'], alpha=0.7)
        axes[1, 0].set_title(f'Prompt Stability (CV < {stable_threshold})')
        axes[1, 0].set_ylabel('Number of Prompts')
        
        # Number of runs per prompt
        axes[1, 1].hist(stability_df['num_runs'], bins=range(1, stability_df['num_runs'].max() + 2),
                       alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Number of Runs per Prompt')
        axes[1, 1].set_xlabel('Number of Runs')
        axes[1, 1].set_ylabel('Number of Prompts')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'stability_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_claim_verification_analysis(self, confidence_df, support_rate):
        """Plot claim verification and confidence analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Support rate pie chart
        support_counts = confidence_df['supported'].value_counts()
        axes[0, 0].pie([support_counts.get(True, 0), support_counts.get(False, 0)], 
                      labels=['Supported', 'Unsupported'], autopct='%1.1f%%',
                      colors=['lightgreen', 'lightcoral'])
        axes[0, 0].set_title(f'Claim Support Distribution\n(Total Claims: {len(confidence_df)})')
        
        # Confidence score distributions
        if len(confidence_df) > 0:
            supported = confidence_df[confidence_df['supported']]
            unsupported = confidence_df[~confidence_df['supported']]
            
            axes[0, 1].hist([supported['max_sim'], unsupported['max_sim']], 
                           bins=20, alpha=0.7, label=['Supported', 'Unsupported'],
                           color=['green', 'red'])
            axes[0, 1].set_title('Max Similarity Score Distribution')
            axes[0, 1].set_xlabel('Max Similarity Score')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            
            # Entailment score comparison
            if len(supported) > 0 and len(unsupported) > 0:
                axes[1, 0].boxplot([supported['entailment'], unsupported['entailment']], 
                                  labels=['Supported', 'Unsupported'])
                axes[1, 0].set_title('Entailment Score Comparison')
                axes[1, 0].set_ylabel('Entailment Score')
            
            # Scatter plot: similarity vs entailment
            scatter = axes[1, 1].scatter(confidence_df['max_sim'], confidence_df['entailment'], 
                                        c=confidence_df['supported'], alpha=0.6,
                                        cmap='RdYlGn')
            axes[1, 1].set_title('Similarity vs Entailment Scores')
            axes[1, 1].set_xlabel('Max Similarity Score')
            axes[1, 1].set_ylabel('Entailment Score')
            plt.colorbar(scatter, ax=axes[1, 1], label='Supported')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'claim_verification_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_runtime_analysis(self, total_runtimes, component_runtimes):
        """Plot runtime analysis charts"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Total runtime distribution
        axes[0, 0].hist(total_runtimes, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Total Runtime Distribution')
        axes[0, 0].set_xlabel('Runtime (seconds)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(total_runtimes), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(total_runtimes):.2f}s')
        axes[0, 0].legend()
        
        # Component runtime comparison (box plot)
        component_data = []
        component_labels = []
        for component, times in component_runtimes.items():
            component_data.append(times)
            component_labels.append(component.replace('_runtime', '').replace('_', ' ').title())
        
        axes[0, 1].boxplot(component_data, labels=component_labels)
        axes[0, 1].set_title('Component Runtime Comparison')
        axes[0, 1].set_ylabel('Runtime (seconds)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Runtime vs experiment index (to check for trends)
        axes[1, 0].plot(range(len(total_runtimes)), total_runtimes, 'o-', alpha=0.6)
        axes[1, 0].set_title('Runtime Over Time')
        axes[1, 0].set_xlabel('Experiment Index')
        axes[1, 0].set_ylabel('Runtime (seconds)')
        
        # Component runtime stacked bar
        component_means = [np.mean(times) for times in component_data]
        axes[1, 1].bar(component_labels, component_means, alpha=0.7)
        axes[1, 1].set_title('Average Component Runtimes')
        axes[1, 1].set_ylabel('Average Runtime (seconds)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'runtime_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_provenance_analysis(self, trace_df):
        """Plot provenance trace quality analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Trace length distribution
        axes[0, 0].hist(trace_df['trace_length'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Trace Length Distribution')
        axes[0, 0].set_xlabel('Number of Trace Steps')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(trace_df['trace_length'].mean(), color='red', linestyle='--',
                          label=f'Mean: {trace_df["trace_length"].mean():.1f}')
        axes[0, 0].legend()
        
        # Facts vs Documents scatter plot
        scatter = axes[0, 1].scatter(trace_df['docs_count'], trace_df['facts_count'], alpha=0.6)
        axes[0, 1].set_title('Facts vs Documents Retrieved')
        axes[0, 1].set_xlabel('Documents Retrieved')
        axes[0, 1].set_ylabel('Facts Extracted')
        
        # Add correlation line
        if len(trace_df) > 1:
            z = np.polyfit(trace_df['docs_count'], trace_df['facts_count'], 1)
            p = np.poly1d(z)
            axes[0, 1].plot(trace_df['docs_count'], p(trace_df['docs_count']), "r--", alpha=0.8)
        
        # Trace completeness rate
        completeness_counts = trace_df['trace_completeness'].value_counts()
        axes[1, 0].bar(['Incomplete', 'Complete'], 
                      [completeness_counts.get(0.0, 0), completeness_counts.get(1.0, 0)],
                      color=['lightcoral', 'lightgreen'], alpha=0.7)
        axes[1, 0].set_title('Trace Completeness')
        axes[1, 0].set_ylabel('Number of Experiments')
        
        # Facts per document ratio distribution
        axes[1, 1].hist(trace_df['facts_per_doc'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Facts per Document Ratio')
        axes[1, 1].set_xlabel('Facts per Document')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(trace_df['facts_per_doc'].mean(), color='red', linestyle='--',
                          label=f'Mean: {trace_df["facts_per_doc"].mean():.2f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'provenance_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    
    # Include all the original methods from MetricsCalculator
    def calculate_system_performance_metrics(self):
        """Calculate system-level performance metrics"""
        t0 = time.time()
        runtimes = []
        component_runtimes = defaultdict(list)
        success_rates = defaultdict(list)
        
        for result in self.results:
            perf = result["system_performance"]
            runtimes.append(perf["total_runtime"])
            
            # Collect component runtimes
            # for component in ["answer_generation_runtime", "alignment_consistency_scoring_runtime",
            #                 "bm25_scoring_sorting_runtime", "document_retrieval_merging_runtime",
            #                 "entity_based_doc_filtering_runtime", "entity_extraction_runtime",
            #                 "loading_resources_runtime", "maximal_span_matching_runtime",
            #                 "tokenize_answer_runtime", "fact_runtime", "halluc_runtime"]:
            for component in ["answer_generation_runtime", "alignment_consistency_scoring_runtime",
                            "entity_based_doc_filtering_runtime", "entity_extraction_runtime",
                            "maximal_span_matching_runtime",
                            "fact_runtime", "halluc_runtime"]:
                if component in perf:
                    component_runtimes[component].append(perf[component])
            
            # Collect success rates
            for component in ["trace_success", "entity_success", "retrieval_success",
                            "fact_success", "halluc_success"]:
                if component in perf:
                    success_rates[component].append(perf[component])
        
        metrics = {
            "runtime_stats": {
                "mean": np.mean(runtimes),
                "median": np.median(runtimes),
                "std": np.std(runtimes),
                "min": np.min(runtimes),
                "max": np.max(runtimes)
            },
            "component_runtime_stats": {
                component: {
                    "mean": np.mean(times),
                    "median": np.median(times),
                    "std": np.std(times)
                } for component, times in component_runtimes.items()
            },
            "success_rates": {
                component: np.mean(successes) 
                for component, successes in success_rates.items()
            }
        }
        
        # Store runtime data for combined plot
        self._runtime_data = {
            "total_runtimes": runtimes,
            "component_runtimes": component_runtimes
        }
        
        # Plot runtime distributions
        self._plot_runtime_analysis(runtimes, component_runtimes)
        
        # Plot success rates
        self._plot_success_rates(success_rates)
        
        calculate_system_performance_time = time.time() - t0
        ic(calculate_system_performance_time)
        return metrics

    # def calculate_system_performance_metrics(self):
    #     """Calculate system-level performance metrics"""
    #     t0 = time.time()
    #     runtimes = []
    #     component_runtimes = defaultdict(list)
    #     success_rates = defaultdict(list)
        
    #     for result in self.results:
    #         perf = result["system_performance"]
    #         runtimes.append(perf["total_runtime"])
            
    #         # Collect component runtimes
    #         for component in ["answer_generation_runtime", "alignment_consistency_scoring_runtime",
    #                           "bm25_scoring_sorting_runtime", "document_retrieval_merging_runtime",
    #                           "entity_based_doc_filtering_runtime", "entity_extraction_runtime",
    #                           "loading_resources_runtime", "maximal_span_matching_runtime",
    #                           "tokenize_answer_runtime", "fact_runtime", "halluc_runtime"]:
    #             if component in perf:
    #                 component_runtimes[component].append(perf[component])
            
    #         # Collect success rates
    #         for component in ["trace_success", "entity_success", "retrieval_success",
    #                           "fact_success", "halluc_success"]:
    #             if component in perf:
    #                 success_rates[component].append(perf[component])
        
    #     metrics = {
    #         "runtime_stats": {
    #             "mean": np.mean(runtimes),
    #             "median": np.median(runtimes),
    #             "std": np.std(runtimes),
    #             "min": np.min(runtimes),
    #             "max": np.max(runtimes)
    #         },
    #         "component_runtime_stats": {
    #             component: {
    #                 "mean": np.mean(times),
    #                 "median": np.median(times),
    #                 "std": np.std(times)
    #             } for component, times in component_runtimes.items()
    #         },
    #         "success_rates": {
    #             component: np.mean(successes) 
    #             for component, successes in success_rates.items()
    #         }
    #     }
        
    #     # Plot runtime distributions
    #     self._plot_runtime_analysis(runtimes, component_runtimes)
        
    #     # Plot success rates
    #     self._plot_success_rates(success_rates)
    #     calculate_system_performance_time = time.time() - t0
    #     ic(calculate_system_performance_time)
    #     return metrics
    
    def calculate_claim_verification_metrics(self):
        """Calculate claim verification and hallucination detection metrics"""
        t0 = time.time()
        all_claims = []
        support_stats = defaultdict(list)
        confidence_scores = []

        for result in tqdm(self.results):
            halluc_results = result["raw_data"]["hallucination_results"]
            
            for hr in halluc_results:
                status = hr.get("status", "").lower()
                all_claims.append(hr)
                
                # Support classification (case-insensitive)
                is_supported = status.startswith("support")
                support_stats["supported"].append(is_supported)
                
                # Confidence scores
                max_sim = hr.get("max_sim", 0)
                avg_sim = hr.get("avg_topk_sim", 0)
                entailment = hr.get("entailment_score", 0)
                
                confidence_scores.append({
                    "max_sim": max_sim,
                    "avg_sim": avg_sim,
                    "entailment": entailment,
                    "supported": is_supported
                })
        
        support_rate = np.mean(support_stats["supported"]) if support_stats["supported"] else 0
        
        # Confidence score analysis
        confidence_df = pd.DataFrame(confidence_scores)
        
        metrics = {
            "total_claims": len(all_claims),
            "support_rate": support_rate,
            "hallucination_rate": 1 - support_rate,
            "confidence_stats": {}
        }
        
        if len(confidence_df) > 0:
            supported_df = confidence_df[confidence_df["supported"]]
            unsupported_df = confidence_df[~confidence_df["supported"]]
            
            metrics["confidence_stats"] = {
                "max_sim": {
                    "supported_mean": supported_df["max_sim"].mean() if len(supported_df) > 0 else 0,
                    "unsupported_mean": unsupported_df["max_sim"].mean() if len(unsupported_df) > 0 else 0
                },
                "entailment": {
                    "supported_mean": supported_df["entailment"].mean() if len(supported_df) > 0 else 0,
                    "unsupported_mean": unsupported_df["entailment"].mean() if len(unsupported_df) > 0 else 0
                }
            }
        
        # Plot claim verification analysis
        if len(confidence_df) > 0:
            self._plot_claim_verification_analysis(confidence_df, support_rate)
        calculate_claim_verification_time = time.time() - t0
        ic(calculate_claim_verification_time)
        return metrics, confidence_df
    
    def calculate_provenance_metrics(self):
        """Calculate provenance trace quality metrics"""
        t0 = time.time()
        trace_stats = []

        for result in tqdm(self.results):
            trace_steps = result["raw_data"]["trace_steps"]
            facts_count = result["outputs"]["facts_count"]
            docs_count = result["outputs"]["matched_docs_count"]
            
            trace_stats.append({
                "trace_length": len(trace_steps),
                "facts_count": facts_count,
                "docs_count": docs_count,
                "facts_per_doc": facts_count / max(docs_count, 1),
                "trace_completeness": 1.0 if facts_count > 0 and docs_count > 0 else 0.0
            })
        
        trace_df = pd.DataFrame(trace_stats)
        
        metrics = {
            "avg_trace_length": trace_df["trace_length"].mean(),
            "avg_facts_per_response": trace_df["facts_count"].mean(),
            "avg_docs_retrieved": trace_df["docs_count"].mean(),
            "trace_completeness_rate": trace_df["trace_completeness"].mean(),
            "facts_to_docs_ratio": trace_df["facts_per_doc"].mean()
        }
        
        # Plot provenance analysis
        self._plot_provenance_analysis(trace_df)
        calculate_provenance_metrics_time = time.time() - t0
        ic(calculate_provenance_metrics_time)
        return metrics, trace_df
    
    def calculate_stability_metrics(self):
        """Calculate output stability across multiple runs"""
        t0 = time.time()
        prompt_groups = defaultdict(list)
        
        # Group results by prompt
        for result in tqdm(self.results):
            prompt_id = result["experiment_metadata"]["prompt_data"]["id"]
            prompt_groups[prompt_id].append(result)
        
        stability_stats = []
        
        for prompt_id, runs in prompt_groups.items():
            if len(runs) < 2:
                continue
                
            # Compare answer lengths and fact counts across runs
            answer_lengths = [len(run["outputs"]["answer"]) for run in runs]
            fact_counts = [run["outputs"]["facts_count"] for run in runs]
            support_rates = []
            
            for run in runs:
                support_rate = run["outputs"]["support_rate"]
                support_rates.append(support_rate)
            
            stability_stats.append({
                "prompt_id": prompt_id,
                "answer_length_cv": np.std(answer_lengths) / np.mean(answer_lengths) if np.mean(answer_lengths) > 0 else 0,
                "fact_count_cv": np.std(fact_counts) / np.mean(fact_counts) if np.mean(fact_counts) > 0 else 0,
                "support_rate_cv": np.std(support_rates) / np.mean(support_rates) if np.mean(support_rates) > 0 else 0,
                "num_runs": len(runs)
            })
        
        stability_df = pd.DataFrame(stability_stats)
        
        if len(stability_df) > 0:
            metrics = {
                "avg_answer_length_cv": stability_df["answer_length_cv"].mean(),
                "avg_fact_count_cv": stability_df["fact_count_cv"].mean(),
                "avg_support_rate_cv": stability_df["support_rate_cv"].mean(),
                "stable_prompts_ratio": (stability_df["support_rate_cv"] < 0.2).mean(),
                "prompts_with_multiple_runs": len(stability_df)
            }
            
            # Plot stability analysis
            self._plot_stability_analysis(stability_df)
            
            # Create combined runtime-stability plot if runtime data is available
            if hasattr(self, '_runtime_data'):
                self._plot_combined_runtime_stability_analysis(
                    self._runtime_data["total_runtimes"],
                    self._runtime_data["component_runtimes"],
                    stability_df
                )
        else:
            metrics = {
                "avg_answer_length_cv": 0,
                "avg_fact_count_cv": 0,
                "avg_support_rate_cv": 0,
                "stable_prompts_ratio": 0,
                "prompts_with_multiple_runs": 0
            }
            
            # Create combined plot even without stability data
            if hasattr(self, '_runtime_data'):
                empty_stability_df = pd.DataFrame()
                self._plot_combined_runtime_stability_analysis(
                    self._runtime_data["total_runtimes"],
                    self._runtime_data["component_runtimes"],
                    empty_stability_df
                )
        
        calculate_stability_metrics_time = time.time() - t0
        ic(calculate_stability_metrics_time)
        return metrics, stability_df

    # def calculate_stability_metrics(self):
    #     """Calculate output stability across multiple runs"""
    #     t0 = time.time()
    #     prompt_groups = defaultdict(list)
        
    #     # Group results by prompt
    #     for result in tqdm(self.results):
    #         prompt_id = result["experiment_metadata"]["prompt_data"]["id"]
    #         prompt_groups[prompt_id].append(result)
        
    #     stability_stats = []
        
    #     for prompt_id, runs in prompt_groups.items():
    #         if len(runs) < 2:
    #             continue
                
    #         # Compare answer lengths and fact counts across runs
    #         answer_lengths = [len(run["outputs"]["answer"]) for run in runs]
    #         fact_counts = [run["outputs"]["facts_count"] for run in runs]
    #         support_rates = []
            
    #         for run in runs:
    #             support_rate = run["outputs"]["support_rate"]
    #             support_rates.append(support_rate)
            
    #         stability_stats.append({
    #             "prompt_id": prompt_id,
    #             "answer_length_cv": np.std(answer_lengths) / np.mean(answer_lengths) if np.mean(answer_lengths) > 0 else 0,
    #             "fact_count_cv": np.std(fact_counts) / np.mean(fact_counts) if np.mean(fact_counts) > 0 else 0,
    #             "support_rate_cv": np.std(support_rates) / np.mean(support_rates) if np.mean(support_rates) > 0 else 0,
    #             "num_runs": len(runs)
    #         })
        
    #     stability_df = pd.DataFrame(stability_stats)
        
    #     if len(stability_df) > 0:
    #         metrics = {
    #             "avg_answer_length_cv": stability_df["answer_length_cv"].mean(),
    #             "avg_fact_count_cv": stability_df["fact_count_cv"].mean(),
    #             "avg_support_rate_cv": stability_df["support_rate_cv"].mean(),
    #             "stable_prompts_ratio": (stability_df["support_rate_cv"] < 0.2).mean(),
    #             "prompts_with_multiple_runs": len(stability_df)
    #         }
            
    #         # Plot stability analysis
    #         self._plot_stability_analysis(stability_df)
    #     else:
    #         metrics = {
    #             "avg_answer_length_cv": 0,
    #             "avg_fact_count_cv": 0,
    #             "avg_support_rate_cv": 0,
    #             "stable_prompts_ratio": 0,
    #             "prompts_with_multiple_runs": 0
    #         }
    #     calculate_stability_metrics_time = time.time() - t0
    #     ic(calculate_stability_metrics_time)
    #     return metrics, stability_df
    
    def _plot_combined_runtime_stability_analysis(self, total_runtimes, component_runtimes, stability_df):
        """Plot combined runtime and stability analysis in 1x3 subplots"""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Set consistent font sizes
        plt.rcParams.update({'font.size': 14})
        
        # Subplot 1: Total runtime distribution
        axes[0].hist(total_runtimes, bins=20, alpha=0.8, edgecolor='black', color='skyblue')
        axes[0].set_title('Total Runtime Distribution', fontsize=16, fontweight='bold')
        axes[0].set_xlabel('Runtime (seconds)', fontsize=14)
        axes[0].set_ylabel('Frequency', fontsize=14)
        axes[0].axvline(np.mean(total_runtimes), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(total_runtimes):.2f}s')
        axes[0].legend(fontsize=12)
        axes[0].tick_params(labelsize=12)
        
        # Subplot 2: Component runtime comparison (box plot)
        if component_runtimes:
            component_data = []
            component_labels = []
            for component, times in component_runtimes.items():
                if len(times) > 0:  # Only include components with data
                    component_data.append(times)
                    component_labels.append(component.replace('_runtime', '').replace('_', ' ').title())
            
            if component_data:
                bp = axes[1].boxplot(component_data, labels=component_labels, patch_artist=True)
                # Color the boxes
                colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                axes[1].set_title('Component Runtime Comparison', fontsize=16, fontweight='bold')
                axes[1].set_ylabel('Runtime (seconds)', fontsize=14)
                axes[1].tick_params(axis='x', rotation=45, labelsize=10)
                axes[1].tick_params(axis='y', labelsize=12)
        else:
            axes[1].text(0.5, 0.5, 'No component runtime data available', 
                        ha='center', va='center', transform=axes[1].transAxes, fontsize=14)
            axes[1].set_title('Component Runtime Comparison', fontsize=16, fontweight='bold')
        
        # Subplot 3: Coefficient of variation distributions from stability analysis
        if len(stability_df) > 0:
            cv_metrics = ['answer_length_cv', 'fact_count_cv', 'support_rate_cv']
            cv_labels = ['Answer Length CV', 'Fact Count CV', 'Support Rate CV']
            colors = ['orange', 'green', 'purple']
            
            # Option 1: Side-by-side box plots (cleaner than overlapping histograms)
            cv_data = []
            valid_labels = []
            for metric, label in zip(cv_metrics, cv_labels):
                if metric in stability_df.columns and len(stability_df[metric]) > 0:
                    cv_data.append(stability_df[metric])
                    valid_labels.append(label)
            
            if cv_data:
                bp = axes[2].boxplot(cv_data, labels=valid_labels, patch_artist=True)
                # Color the boxes
                box_colors = ['orange', 'green', 'purple'][:len(bp['boxes'])]
                for patch, color in zip(bp['boxes'], box_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                axes[2].set_title('Coefficient of Variation Distributions', fontsize=16, fontweight='bold')
                axes[2].set_ylabel('Coefficient of Variation', fontsize=14)
                axes[2].tick_params(axis='x', rotation=45, labelsize=12)
                axes[2].tick_params(axis='y', labelsize=12)
                axes[2].grid(True, alpha=0.3)
              
        else:
            axes[2].text(0.5, 0.5, 'No stability data available\n(Need multiple runs per prompt)', 
                        ha='center', va='center', transform=axes[2].transAxes, fontsize=14)
            axes[2].set_title('Coefficient of Variation Distributions', fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'combined_runtime_stability_analysis.png'), 
                    dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.plots_dir, 'combined_runtime_stability_analysis.pdf'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Reset font size to default
        plt.rcParams.update({'font.size': plt.rcParamsDefault['font.size']})
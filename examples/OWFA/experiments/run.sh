#!bin/bash

# Quick test with 2 prompts, 2 runs each
python run_experiments.py --experiment-name quick_test --max-prompts 5 --runs-per-prompt 2

# Full experiment
python run_experiments.py --experiment-name full_analysis --full --runs-per-prompt 3

# Custom prompt file
python run_experiments.py --prompts-file custom_prompts.json --experiment-name custom_experiment

# run this
python metrics_calculator.py --results_dir "experiment_data/experiment_results" --experiment_name "quick_test" --run_evaluation "general_analysis"
python metrics_calculator.py --results_dir "experiment_data/experiment_results" --experiment_name "full_analysis" --run_evaluation "general_analysis"

# run this
python metrics_calculator.py --results_dir "experiment_data/experiment_results" --experiment_name "quick_test" --run_evaluation "hallucination_detection"
python metrics_calculator.py --results_dir "experiment_data/experiment_results" --experiment_name "full_analysis" --run_evaluation "hallucination_detection"

# run this
python metrics_calculator.py --results_dir "experiment_data/experiment_results" --experiment_name "quick_test" --run_evaluation "hit_at_5"
python metrics_calculator.py --results_dir "experiment_data/experiment_results" --experiment_name "full_analysis" --run_evaluation "hit_at_5"

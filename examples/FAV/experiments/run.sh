#!/bin/bash


python comprehensive_experiment_runner.py --experiment-name "test_experiment" --max-prompts 2 --runs-per-prompt 2

python comprehensive_experiment_runner.py --experiment-name "pilot_experiment" --full --runs-per-prompt 2

python metrics_calculator.py --results_dir "experiment_data/experiment_results" --experiment_name "pilot_experiment" --run_evaluation "general_analysis"

python metrics_calculator.py --results_dir "experiment_data/experiment_results" --experiment_name "pilot_experiment" --run_evaluation "hallucination_detection"

python metrics_calculator.py --results_dir "experiment_data/experiment_results" --experiment_name "pilot_experiment" --run_evaluation "hit_at_5"
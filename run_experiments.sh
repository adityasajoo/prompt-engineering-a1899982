#!/bin/bash

# Prompt Optimization Framework - Run Scripts
# For MacBook M4 with 16GB RAM

echo "=================================================="
echo "Prompt Optimization Framework"
echo "=================================================="

# Function to display menu
show_menu() {
    echo ""
    echo "Select an option:"
    echo "1) Quick Test (1 model, 1 dataset, 5 generations)"
    echo "2) Single Model - All Datasets"
    echo "3) All Models - Single Dataset"
    echo "4) Full Experiment (All models, All datasets)"
    echo "5) Memory-Optimized Run (for 16GB RAM)"
    echo "6) Custom Configuration"
    echo "7) Generate Report from Existing Results"
    echo "8) Exit"
    echo ""
}

# Quick test run
quick_test() {
    echo "Running quick test..."
    python prompt_optimizer.py \
        --mode single \
        --model "google/flan-t5-base" \
        --dataset "gsm8k" \
        --num-generations 5 \
        --population-size 10
}

# Single model, all datasets
single_model_all_datasets() {
    echo "Enter model name (e.g., google/flan-t5-base):"
    read model_name
    
    for dataset in "gsm8k" "truthful_qa" "mmlu"; do
        echo "Running $model_name on $dataset..."
        python prompt_optimizer.py \
            --mode single \
            --model "$model_name" \
            --dataset "$dataset" \
            --num-generations 8 \
            --population-size 15
    done
}

# All models, single dataset
all_models_single_dataset() {
    echo "Enter dataset name (gsm8k, truthful_qa, or mmlu):"
    read dataset_name
    
    # Using smaller models for memory efficiency
    for model in "google/flan-t5-base" "microsoft/phi-2"; do
        echo "Running $model on $dataset_name..."
        python prompt_optimizer.py \
            --mode single \
            --model "$model" \
            --dataset "$dataset_name" \
            --num-generations 8 \
            --population-size 15
    done
}

# Full experiment
full_experiment() {
    echo "WARNING: This will run all models on all datasets."
    echo "This may take several hours and require significant memory."
    echo "Continue? (y/n)"
    read confirm
    
    if [ "$confirm" = "y" ]; then
        python prompt_optimizer.py --mode all
    else
        echo "Cancelled."
    fi
}

# Memory-optimized run for 16GB RAM
memory_optimized() {
    echo "Running memory-optimized configuration..."
    python prompt_optimizer.py \
        --mode all \
        --population-size 8 \
        --num-generations 5
}

# Custom configuration
custom_config() {
    echo "Enter model name:"
    read model
    echo "Enter dataset name:"
    read dataset
    echo "Enter population size (recommended: 10-20):"
    read pop_size
    echo "Enter number of generations (recommended: 5-15):"
    read num_gen
    echo "Enter mutation rate (recommended: 0.1-0.3):"
    read mut_rate
    
    python prompt_optimizer.py \
        --mode single \
        --model "$model" \
        --dataset "$dataset" \
        --population-size "$pop_size" \
        --num-generations "$num_gen" \
        --mutation-rate "$mut_rate"
}

# Generate report from existing results
generate_report() {
    echo "Generating report from existing results..."
    python -c "
import os
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Find most recent results directory
results_dir = 'prompt_optimization_results'
if os.path.exists(results_dir):
    dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    if dirs:
        latest_dir = sorted(dirs)[-1]
        results_path = os.path.join(results_dir, latest_dir)
        
        # Load results
        csv_path = os.path.join(results_path, 'results_summary.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            # Generate summary statistics
            print('\n' + '='*60)
            print('RESULTS SUMMARY')
            print('='*60)
            print(f'Total Evaluations: {len(df)}')
            print(f'Best Fitness: {df[\"fitness\"].max():.4f}')
            print(f'Average Fitness: {df[\"fitness\"].mean():.4f}')
            
            print('\n--- By Model ---')
            print(df.groupby('model')['fitness'].agg(['mean', 'max', 'std']))
            
            print('\n--- By Dataset ---')
            print(df.groupby('dataset')['fitness'].agg(['mean', 'max', 'std']))
            
            print('\n--- By Strategy ---')
            print(df.groupby('strategy')['fitness'].agg(['mean', 'max', 'count']))
            
            print(f'\nFull results available at: {results_path}')
        else:
            print('No results CSV found.')
    else:
        print('No results directories found.')
else:
    print('Results directory not found.')
"
}

# Main loop
while true; do
    show_menu
    read -p "Enter choice [1-8]: " choice
    
    case $choice in
        1)
            quick_test
            ;;
        2)
            single_model_all_datasets
            ;;
        3)
            all_models_single_dataset
            ;;
        4)
            full_experiment
            ;;
        5)
            memory_optimized
            ;;
        6)
            custom_config
            ;;
        7)
            generate_report
            ;;
        8)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid option. Please try again."
            ;;
    esac
    
    echo ""
    echo "Press Enter to continue..."
    read
done
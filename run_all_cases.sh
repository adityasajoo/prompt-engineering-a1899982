#!/bin/bash

echo "=========================================="
echo "Running All Cases - Prompt Optimization"
echo "=========================================="

MODELS=("gpt2" "google/flan-t5-small" "facebook/opt-125m")
DATASETS=("cnn_dailymail" "samsum" "xsum")

# Configuration
GENERATIONS=8
POP_SIZE=10

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "=========================================="
    echo "CASE: Dataset = $dataset"
    echo "=========================================="
    
    for model in "${MODELS[@]}"; do
        echo ""
        echo "------------------------------------------"
        echo "Running: $model on $dataset"
        echo "------------------------------------------"
        
        python3 prompt_optimizer.py \
            --mode single \
            --model "$model" \
            --dataset "$dataset" \
            --num-generations $GENERATIONS \
            --population-size $POP_SIZE \
            --mutation-rate 0.15
        
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to run $model on $dataset"
            echo "Continuing with next configuration..."
        else
            echo "SUCCESS: Completed $model on $dataset"
        fi
        
        # Brief pause to let system recover
        sleep 5
    done
done

echo ""
echo "=========================================="
echo "All cases completed!"
echo "=========================================="
echo "Generating comprehensive report..."

python3 generate_comprehensive_report.py

echo ""
echo "Done! Check comprehensive_report.md for results"

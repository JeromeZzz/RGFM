#!/bin/bash

# KumoRFM RelBench Experiment Running Script
# Used for batch running multiple experiments

# Set base parameters
BASE_DIR="./relbench_outputs"
LOG_DIR="./logs"
DEVICE="cuda"

# Create directories
mkdir -p $BASE_DIR
mkdir -p $LOG_DIR

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print info
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Experiment 1: Amazon User Churn Prediction (Base Experiment)
run_amazon_churn() {
    print_info "Running Experiment: Amazon User Churn Prediction"
    
    python relbench/train_on_relbench.py \
        --dataset amazon \
        --task user-churn \
        --hidden-dim 256 \
        --num-layers 4 \
        --num-heads 8 \
        --dropout 0.3 \
        --epochs 30 \
        --batch-size 64 \
        --lr 0.0001 \
        --device $DEVICE \
        --output-dir $BASE_DIR \
        2>&1 | tee $LOG_DIR/amazon_churn_$(date +%Y%m%d_%H%M%S).log
}

# Experiment 2: Stack User Badge Prediction (Multi-class Classification)
run_stack_badge() {
    print_info "Running Experiment: Stack User Badge Prediction"
    
    python relbench/train_on_relbench.py \
        --dataset stack \
        --task user-badge \
        --hidden-dim 256 \
        --num-layers 4 \
        --num-heads 8 \
        --dropout 0.3 \
        --epochs 40 \
        --batch-size 32 \
        --lr 0.0001 \
        --device $DEVICE \
        --output-dir $BASE_DIR \
        2>&1 | tee $LOG_DIR/stack_badge_$(date +%Y%m%d_%H%M%S).log
}

# Experiment 3: F1 Driver Position Prediction (Regression)
run_f1_position() {
    print_info "Running Experiment: F1 Driver Position Prediction"
    
    python relbench/train_on_relbench.py \
        --dataset f1 \
        --task driver-position \
        --hidden-dim 256 \
        --num-layers 4 \
        --num-heads 8 \
        --dropout 0.2 \
        --epochs 50 \
        --batch-size 128 \
        --lr 0.0005 \
        --device $DEVICE \
        --output-dir $BASE_DIR \
        2>&1 | tee $LOG_DIR/f1_position_$(date +%Y%m%d_%H%M%S).log
}

# Hyperparameter search experiment
run_hyperparam_search() {
    print_info "Running Hyperparameter Search Experiment"
    
    dataset="amazon"
    task="user-churn"
    
    for hidden_dim in 128 256 512; do
        for num_layers in 2 3 4; do
            for lr in 0.0001 0.0005; do
                print_info "Hyperparameters: hidden_dim=$hidden_dim, num_layers=$num_layers, lr=$lr"
                
                python relbench/train_on_relbench.py \
                    --dataset $dataset \
                    --task $task \
                    --hidden-dim $hidden_dim \
                    --num-layers $num_layers \
                    --num-heads 4 \
                    --dropout 0.3 \
                    --epochs 20 \
                    --batch-size 64 \
                    --lr $lr \
                    --device $DEVICE \
                    --output-dir $BASE_DIR/hp_search \
                    2>&1 | tee $LOG_DIR/hp_search_${hidden_dim}_${num_layers}_${lr}_$(date +%Y%m%d_%H%M%S).log
            done
        done
    done
}

# Quick test (for environment validation)
run_quick_test() {
    print_info "Running Quick Test"
    
    python relbench/train_on_relbench.py \
        --dataset amazon \
        --task user-churn \
        --hidden-dim 64 \
        --num-layers 1 \
        --num-heads 2 \
        --dropout 0.1 \
        --epochs 2 \
        --batch-size 16 \
        --lr 0.001 \
        --device $DEVICE \
        --output-dir $BASE_DIR/test \
        2>&1 | tee $LOG_DIR/quick_test_$(date +%Y%m%d_%H%M%S).log
}

# Run all benchmark experiments
run_all_benchmarks() {
    print_info "Running All Benchmark Experiments"
    
    # Amazon dataset
    for task in user-churn user-ltv item-churn item-ltv; do
        print_info "Running: Amazon $task"
        python relbench/train_on_relbench.py \
            --dataset amazon \
            --task $task \
            --hidden-dim 256 \
            --num-layers 4 \
            --device $DEVICE \
            --output-dir $BASE_DIR \
            2>&1 | tee $LOG_DIR/amazon_${task}_$(date +%Y%m%d_%H%M%S).log
    done
    
    # Stack dataset
    for task in user-badge user-engagement post-votes; do
        print_info "Running: Stack $task"
        python relbench/train_on_relbench.py \
            --dataset stack \
            --task $task \
            --hidden-dim 256 \
            --num-layers 4 \
            --device $DEVICE \
            --output-dir $BASE_DIR \
            2>&1 | tee $LOG_DIR/stack_${task}_$(date +%Y%m%d_%H%M%S).log
    done
}

# Analyze results
analyze_results() {
    print_info "Analyzing Experiment Results"
    
    python relbench/analyze_results.py \
        --results-dir $BASE_DIR \
        --output-dir ./analysis \
        2>&1 | tee $LOG_DIR/analysis_$(date +%Y%m%d_%H%M%S).log
}

# Main menu
show_menu() {
    echo ""
    echo "KumoRFM RelBench Experiment Menu"
    echo "========================="
    echo "1. Run Amazon User Churn Prediction"
    echo "2. Run Stack User Badge Prediction"
    echo "3. Run F1 Driver Position Prediction"
    echo "4. Run Hyperparameter Search"
    echo "5. Run Quick Test"
    echo "6. Run All Benchmark Experiments"
    echo "7. Analyze Experiment Results"
    echo "8. Exit"
    echo ""
}

# Main loop
while true; do
    show_menu
    read -p "Please select an operation (1-8): " choice
    
    case $choice in
        1)
            run_amazon_churn
            ;;
        2)
            run_stack_badge
            ;;
        3)
            run_f1_position
            ;;
        4)
            run_hyperparam_search
            ;;
        5)
            run_quick_test
            ;;
        6)
            run_all_benchmarks
            ;;
        7)
            analyze_results
            ;;
        8)
            print_info "Exiting program"
            exit 0
            ;;
        *)
            print_error "Invalid choice, please re-enter"
            ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
done
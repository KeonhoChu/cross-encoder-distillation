#!/bin/bash

# Cross-Encoder to Bi-Encoder Knowledge Distillation Pipeline Runner
# Runs the full training pipeline: Teacher Training -> Student Training -> Evaluation.
# Usage: ./run_pipeline.sh [quick|full] [gpu_device]
# Default mode is 'quick', default gpu is '7'.

MODE=${1:-quick}
GPU_DEVICE=${2:-7}
PYTHON="python" # Or path to specific python executable

# Set GPU Device
export CUDA_VISIBLE_DEVICES=$GPU_DEVICE

# Colors
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}Starting Pipeline in [${MODE}] mode on GPU [${GPU_DEVICE}]${NC}"
echo -e "${CYAN}============================================================${NC}"

# Debug: Check GPU visibility
echo -e "\n${YELLOW}[Debug] Checking GPU visibility (nvidia-smi):${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not found"
fi
echo -e "${YELLOW}[Debug] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES${NC}\n"

if [ "$MODE" == "quick" ]; then
    # --- Quick Mode Configuration ---
    TEACHER_DIR="./models/cross_encoder_teacher_quick"
    STUDENT_DIR="./models/bi_encoder_distilled_quick"
    OUTPUT_DIR="./evaluation_results/quick_test"
    
    # 1. Teacher Training
    echo -e "\n${YELLOW}[Step 1] Training Teacher (Quick)...${NC}"
    $PYTHON train_teacher_model.py \
        --data_version supervised_only \
        --output_dir "$TEACHER_DIR" \
        --epochs 1 \
        --batch_size 32 \
        --lr 2e-5 \
        --use_lora \
        --lora_r 16 \
        --lora_alpha 32 \
        --max_samples 2000

    # 2. Student Training
    echo -e "\n${YELLOW}[Step 2] Training Student (Quick)...${NC}"
    $PYTHON train_distillation_pipeline.py \
        --teacher_path "$TEACHER_DIR" \
        --student_name intfloat/multilingual-e5-large-instruct \
        --data_version supervised_only \
        --output_dir "$STUDENT_DIR" \
        --epochs 1 \
        --batch_size 16 \
        --lr 2e-5 \
        --K 4 \
        --tau 0.2 \
        --contrastive_weight 0.3 \
        --negative_penalty_weight 0.4 \
        --use_lora \
        --max_len 128 \
        --max_samples 2000

elif [ "$MODE" == "full" ]; then
    # --- Full Mode Configuration ---
    TEACHER_DIR="./models/cross_encoder_teacher"
    STUDENT_DIR="./models/bi_encoder_distilled_improved"
    OUTPUT_DIR="./evaluation_results/final_model_normalized"

    # 1. Teacher Training
    echo -e "\n${YELLOW}[Step 1] Training Teacher (Full)...${NC}"
    $PYTHON train_teacher_model.py \
        --data_version supervised_only \
        --output_dir "$TEACHER_DIR" \
        --epochs 3 \
        --batch_size 16 \
        --lr 2e-5 \
        --use_lora \
        --lora_r 32 \
        --lora_alpha 64 \
        --lora_dropout 0.1

    # 2. Student Training
    echo -e "\n${YELLOW}[Step 2] Training Student (Full)...${NC}"
    $PYTHON train_distillation_pipeline.py \
        --teacher_path "$TEACHER_DIR" \
        --student_name intfloat/multilingual-e5-large-instruct \
        --data_version supervised_only \
        --output_dir "$STUDENT_DIR" \
        --epochs 3 \
        --batch_size 4 \
        --lr 2e-5 \
        --K 8 \
        --tau 0.2 \
        --contrastive_weight 0.3 \
        --negative_penalty_weight 0.4 \
        --use_lora \
        --lora_r 32 \
        --lora_alpha 64 \
        --lora_dropout 0.1

else
    echo "Invalid mode: $MODE. Use 'quick' or 'full'."
    exit 1
fi

# 3. Evaluation (Common)
echo -e "\n${YELLOW}[Step 3] Evaluating Model...${NC}"
$PYTHON evaluate_final_model.py \
    --model_path "$STUDENT_DIR" \
    --test_data ./preprocessed/supervised_only/test.csv \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 8

echo -e "\n${CYAN}============================================================${NC}"
echo -e "${CYAN}Pipeline Completed Successfully!${NC}"
echo -e "${CYAN}Results saved to: $OUTPUT_DIR${NC}"
echo -e "${CYAN}============================================================${NC}"

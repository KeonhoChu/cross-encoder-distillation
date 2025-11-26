<#
.SYNOPSIS
    Cross-Encoder to Bi-Encoder Knowledge Distillation Pipeline Runner
.DESCRIPTION
    Runs the full training pipeline: Teacher Training -> Student Training -> Evaluation.
    Supports 'quick' mode for testing and 'full' mode for production training.
.PARAMETER Mode
    'quick' or 'full'. Default is 'quick'.
.EXAMPLE
    .\run_pipeline.ps1 -Mode quick
.EXAMPLE
    .\run_pipeline.ps1 -Mode full
#>

param (
    [ValidateSet("quick", "full")]
    [string]$Mode = "quick",

    [string]$GpuDevice = "0"
)

$ErrorActionPreference = "Stop"

# Set GPU Device
$env:CUDA_VISIBLE_DEVICES = $GpuDevice

# Configuration
$BaseDir = Get-Location
$Python = "python" # Or path to specific python executable

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Starting Pipeline in [$Mode] mode on GPU [$GpuDevice]" -ForegroundColor Cyan
Write-Host "============================================================"

if ($Mode -eq "quick") {
    # --- Quick Mode Configuration ---
    $TeacherDir = "./models/cross_encoder_teacher_quick"
    $StudentDir = "./models/bi_encoder_distilled_quick"
    $OutputDir = "./evaluation_results/quick_test"
    
    # 1. Teacher Training
    Write-Host "`n[Step 1] Training Teacher (Quick)..." -ForegroundColor Yellow
    & $Python train_teacher_model.py `
        --data_version supervised_only `
        --output_dir $TeacherDir `
        --epochs 1 `
        --batch_size 32 `
        --lr 2e-5 `
        --use_lora `
        --lora_r 16 `
        --lora_alpha 32 `
        --max_samples 2000

    # 2. Student Training
    Write-Host "`n[Step 2] Training Student (Quick)..." -ForegroundColor Yellow
    & $Python train_distillation_pipeline.py `
        --teacher_path $TeacherDir `
        --student_name intfloat/multilingual-e5-large-instruct `
        --data_version supervised_only `
        --output_dir $StudentDir `
        --epochs 1 `
        --batch_size 16 `
        --lr 2e-5 `
        --K 4 `
        --tau 0.2 `
        --contrastive_weight 0.3 `
        --negative_penalty_weight 0.4 `
        --use_lora `
        --max_len 128 `
        --max_samples 2000

} else {
    # --- Full Mode Configuration ---
    $TeacherDir = "./models/cross_encoder_teacher"
    $StudentDir = "./models/bi_encoder_distilled_improved"
    $OutputDir = "./evaluation_results/final_model_normalized"

    # 1. Teacher Training
    Write-Host "`n[Step 1] Training Teacher (Full)..." -ForegroundColor Yellow
    & $Python train_teacher_model.py `
        --data_version supervised_only `
        --output_dir $TeacherDir `
        --epochs 3 `
        --batch_size 16 `
        --lr 2e-5 `
        --use_lora `
        --lora_r 32 `
        --lora_alpha 64 `
        --lora_dropout 0.1

    # 2. Student Training
    Write-Host "`n[Step 2] Training Student (Full)..." -ForegroundColor Yellow
    & $Python train_distillation_pipeline.py `
        --teacher_path $TeacherDir `
        --student_name intfloat/multilingual-e5-large-instruct `
        --data_version supervised_only `
        --output_dir $StudentDir `
        --epochs 3 `
        --batch_size 4 `
        --lr 2e-5 `
        --K 8 `
        --tau 0.2 `
        --contrastive_weight 0.3 `
        --negative_penalty_weight 0.4 `
        --use_lora `
        --lora_r 32 `
        --lora_alpha 64 `
        --lora_dropout 0.1
}

# 3. Evaluation (Common)
Write-Host "`n[Step 3] Evaluating Model..." -ForegroundColor Yellow
& $Python evaluate_final_model.py `
    --model_path $StudentDir `
    --test_data ./preprocessed/supervised_only/test.csv `
    --output_dir $OutputDir `
    --batch_size 8

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "Pipeline Completed Successfully!" -ForegroundColor Cyan
Write-Host "Results saved to: $OutputDir" -ForegroundColor Cyan
Write-Host "============================================================"

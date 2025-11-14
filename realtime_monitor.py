#!/usr/bin/env python3
"""
실시간 학습 모니터링 및 자동 복구 스크립트
- 학습 프로세스 상태 실시간 확인
- 에러 발생 시 자동 감지 및 복구
- 학습 완료 시 다음 단계 자동 실행
"""

import subprocess
import time
import os
import re
import sys
import signal

# 모니터링 설정
CHECK_INTERVAL = 30  # 30초마다 체크
LOG_FILE = "/embedding/finetuning_embedding/train_improved_v2.log"
TRAINING_SCRIPT = "/embedding/finetuning_embedding/train_distillation_pipeline.py"
EVALUATION_SCRIPT = "/embedding/finetuning_embedding/evaluate_final_model.py"

# 학습 파라미터 (에러 복구용)
TRAINING_ARGS = {
    "teacher_path": "./models/cross_encoder_teacher",
    "student_name": "intfloat/multilingual-e5-large-instruct",
    "data_version": "supervised_only",
    "output_dir": "./models/bi_encoder_distilled_improved",
    "epochs": "3",
    "batch_size": "2",  # GPU 메모리 문제 대응
    "lr": "2e-5",
    "K": "8",  # GPU 메모리 문제 대응
    "tau": "0.2",
    "use_hard_negative_mining": "True",
    "normalize_similarity": "True",
    "use_lora": "",
    "lora_r": "32",
    "lora_alpha": "64",
    "lora_dropout": "0.1",
    "max_len": "256",
    "device": "cuda"
}

# 복구 전략: 에러 타입별 파라미터 조정
RECOVERY_STRATEGIES = {
    "gpu_memory": {"batch_size": "1", "K": "4"},
    "cuda": {"device": "cpu"},  # 최후의 수단
}

def check_process_running(process_name="train_distillation"):
    """학습 프로세스가 실행 중인지 확인"""
    try:
        result = subprocess.run(
            ["pgrep", "-f", process_name],
            capture_output=True,
            text=True
        )
        return result.returncode == 0 and result.stdout.strip() != ""
    except:
        return False

def check_training_complete(log_file):
    """로그 파일에서 학습 완료 여부 확인"""
    if not os.path.exists(log_file):
        return False
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            # 마지막 100줄만 읽기
            lines = f.readlines()[-100:]
            content = ''.join(lines).lower()
            
            # 학습 완료 표시
            completion_indicators = [
                "모델 저장 완료",
                "학습 완료",
                "training completed",
                "모델이 저장되었습니다",
                "saved model",
                "epoch 3/3",
                "100%|██████████|"
            ]
            
            for indicator in completion_indicators:
                if indicator.lower() in content:
                    return True
            
            # 마지막 로그에서 100% 또는 완료 메시지 확인
            last_lines = lines[-10:]
            for line in last_lines:
                if "100%" in line or "완료" in line or "complete" in line.lower():
                    return True
            
            return False
    except Exception as e:
        print(f"로그 파일 읽기 오류: {e}")
        return False

def detect_error(log_file):
    """로그 파일에서 에러 패턴 감지"""
    if not os.path.exists(log_file):
        return None, None
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            # 마지막 200줄만 읽기
            lines = f.readlines()[-200:]
            content = ''.join(lines)
            
            # GPU 메모리 에러
            if any(pattern in content for pattern in [
                "out of memory", "cuda out of memory", "nvml",
                "runtimeerror: nvml_success", "cuda error"
            ]):
                return "gpu_memory", "GPU 메모리 부족 에러 감지"
            
            # CUDA 에러
            if any(pattern in content for pattern in [
                "cuda runtime error", "cuda driver error", "cuda initialization error"
            ]):
                return "cuda", "CUDA 에러 감지"
            
            # Import 에러
            if "moduleNotFoundError" in content or "import error" in content.lower():
                return "import", "Import 에러 감지"
            
            # 일반 에러
            if "error" in content.lower() or "exception" in content.lower() or "traceback" in content.lower():
                # 최근 에러 메시지 찾기
                for line in reversed(lines[-50:]):
                    if "error" in line.lower() or "exception" in line.lower():
                        return "general", f"에러 감지: {line.strip()[:100]}"
            
            return None, None
    except Exception as e:
        print(f"에러 감지 중 오류: {e}")
        return None, None

def kill_training_process():
    """실행 중인 학습 프로세스 종료"""
    try:
        subprocess.run(["pkill", "-f", "train_distillation"], check=False)
        time.sleep(2)
        # GPU 캐시 정리
        subprocess.run([
            "python", "-c",
            "import torch; torch.cuda.empty_cache(); print('GPU 캐시 정리 완료')"
        ], cwd="/embedding/finetuning_embedding", check=False)
    except:
        pass

def recover_from_error(error_type, error_msg):
    """에러 타입에 따라 복구 전략 적용"""
    print(f"\n[복구 시작] {error_type}: {error_msg}")
    
    # 실행 중인 프로세스 종료
    kill_training_process()
    
    # 복구 전략 적용
    recovery_params = RECOVERY_STRATEGIES.get(error_type, {})
    
    # 학습 파라미터 업데이트
    current_params = TRAINING_ARGS.copy()
    current_params.update(recovery_params)
    
    print(f"[복구] 파라미터 조정: {recovery_params}")
    
    # 학습 재시작
    restart_training(current_params)

def restart_training(params=None):
    """학습 재시작"""
    if params is None:
        params = TRAINING_ARGS
    
    print(f"\n[학습 재시작] 파라미터: {params}")
    
    # 명령어 구성
    cmd = [
        "python", TRAINING_SCRIPT,
        "--teacher_path", params["teacher_path"],
        "--student_name", params["student_name"],
        "--data_version", params["data_version"],
        "--output_dir", params["output_dir"],
        "--epochs", params["epochs"],
        "--batch_size", params["batch_size"],
        "--lr", params["lr"],
        "--K", params["K"],
        "--tau", params["tau"],
        "--use_hard_negative_mining", params["use_hard_negative_mining"],
        "--normalize_similarity", params["normalize_similarity"],
        "--lora_r", params["lora_r"],
        "--lora_alpha", params["lora_alpha"],
        "--lora_dropout", params["lora_dropout"],
        "--max_len", params["max_len"],
        "--device", params["device"]
    ]
    
    if params.get("use_lora"):
        cmd.append("--use_lora")
    
    # 백그라운드 실행 및 로그 저장
    log_path = f"{LOG_FILE}.recovered.{int(time.time())}"
    with open(log_path, 'w') as log_file:
        process = subprocess.Popen(
            cmd,
            cwd="/embedding/finetuning_embedding",
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
    
    print(f"[학습 재시작 완료] PID: {process.pid}, 로그: {log_path}")

def run_evaluation():
    """학습 완료 후 평가 실행"""
    print("\n[다음 단계] 모델 평가 시작")
    
    cmd = ["python", EVALUATION_SCRIPT]
    
    try:
        result = subprocess.run(
            cmd,
            cwd="/embedding/finetuning_embedding",
            capture_output=True,
            text=True,
            timeout=3600  # 1시간 타임아웃
        )
        print(f"[평가 완료] 코드: {result.returncode}")
        if result.stdout:
            print(f"[평가 출력]\n{result.stdout}")
        if result.stderr:
            print(f"[평가 에러]\n{result.stderr}")
    except subprocess.TimeoutExpired:
        print("[평가] 타임아웃 (1시간 초과)")
    except Exception as e:
        print(f"[평가 오류] {e}")

def monitor():
    """실시간 모니터링 메인 루프"""
    print("=" * 60)
    print("실시간 학습 모니터링 시작")
    print(f"로그 파일: {LOG_FILE}")
    print(f"체크 간격: {CHECK_INTERVAL}초")
    print("=" * 60)
    
    consecutive_errors = 0
    max_consecutive_errors = 3
    
    while True:
        try:
            # 프로세스 실행 상태 확인
            is_running = check_process_running()
            
            # 학습 완료 확인
            is_complete = check_training_complete(LOG_FILE)
            
            # 에러 감지
            error_type, error_msg = detect_error(LOG_FILE)
            
            # 상태 출력
            status_msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
            if is_complete:
                status_msg += "✓ 학습 완료"
                print(status_msg)
                print("[다음 단계] 평가 스크립트 실행")
                run_evaluation()
                print("[모니터링 종료] 모든 작업 완료")
                break
            elif error_type:
                consecutive_errors += 1
                status_msg += f"✗ 에러 감지 ({consecutive_errors}/{max_consecutive_errors}): {error_type}"
                print(status_msg)
                
                if consecutive_errors >= max_consecutive_errors:
                    print(f"[중단] 연속 에러 {max_consecutive_errors}회 발생. 수동 확인 필요.")
                    break
                
                # 에러 복구 시도
                recover_from_error(error_type, error_msg)
                consecutive_errors = 0  # 복구 시도 후 리셋
            elif is_running:
                # 진행률 확인
                try:
                    with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-1].strip()
                            if "Training:" in last_line:
                                status_msg += f"▶ 진행 중: {last_line[-80:]}"
                            else:
                                status_msg += f"▶ 실행 중 (PID 확인됨)"
                except:
                    status_msg += "▶ 실행 중"
                
                print(status_msg)
                consecutive_errors = 0  # 정상 실행 중이면 에러 카운트 리셋
            else:
                status_msg += "⚠ 프로세스 없음 (완료 또는 중단)"
                print(status_msg)
                
                # 프로세스가 없는데 완료도 아닌 경우
                if not is_complete:
                    print("[재시작] 프로세스가 없어서 재시작 시도")
                    restart_training()
            
            # 대기
            time.sleep(CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n[모니터링 중단] 사용자 요청")
            break
        except Exception as e:
            print(f"[모니터링 오류] {e}")
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    monitor()









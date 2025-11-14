#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
supervised CSV 파일들을 합치고 train/val/test로 분할하는 통합 스크립트
"""

import pandas as pd
import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split

def merge_supervised_csv_files():
    """supervised CSV 파일들을 합치는 함수"""
    # 소스 폴더와 대상 폴더 경로
    source_folder = "/embedding/preprocessed/results_v2/"
    target_folder = "/embedding/ver_2_e5_large_instruct/preprocessed/"
    detailed_file = "/embedding/multi_lora_domain_training_v2/detailed_similarity_results_v2_top20.csv"
    
    # supervised CSV 파일들 찾기
    pattern = os.path.join(source_folder, "*_supervised.csv")
    csv_files = glob.glob(pattern)
    
    print(f"발견된 supervised CSV 파일들:")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")
    
    if not csv_files:
        print("supervised CSV 파일을 찾을 수 없습니다.")
        return None, None
    
    # 모든 CSV 파일을 읽어서 합치기
    dataframes = []
    for file in csv_files:
        print(f"읽는 중: {os.path.basename(file)}")
        df = pd.read_csv(file)
        dataframes.append(df)
    
    # supervised 데이터프레임 합치기
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    # negative 컬럼들을 하나만 남기고 나머지 제거 (negative_1만 남기고 negative로 이름 변경)
    columns_to_keep = ['anchor', 'positive', 'negative_1']
    merged_df = merged_df[columns_to_keep].copy()
    merged_df = merged_df.rename(columns={'negative_1': 'negative'})
    
    # detailed_similarity_results_v2_top20.csv 파일 읽기
    print(f"\n읽는 중: detailed_similarity_results_v2_top20.csv")
    detailed_df = pd.read_csv(detailed_file)
    
    # 필요한 컬럼만 선택하고 컬럼명 변경
    detailed_columns = ['anchor', 'original_positive', 'best_other_positive']
    detailed_df = detailed_df[detailed_columns].copy()
    detailed_df = detailed_df.rename(columns={
        'original_positive': 'positive',
        'best_other_positive': 'negative'
    })
    
    # 두 데이터프레임 합치기
    final_df = pd.concat([merged_df, detailed_df], ignore_index=True)
    
    print(f"\n합치기 완료!")
    print(f"supervised 데이터 행 수: {len(merged_df)}")
    print(f"detailed 데이터 행 수: {len(detailed_df)}")
    print(f"총 행 수: {len(final_df)}")
    print(f"총 열 수: {len(final_df.columns)}")
    
    # 각 원본 파일의 행 수 출력
    print(f"\n각 파일별 행 수:")
    for i, file in enumerate(csv_files):
        print(f"  {os.path.basename(file)}: {len(dataframes[i])} 행")
    print(f"  detailed_similarity_results_v2_top20.csv: {len(detailed_df)} 행")
    
    return merged_df, final_df

def split_data(supervised_df, combined_df):
    """데이터를 train/val/test로 분할하는 함수"""
    base_folder = "/embedding/ver_2_e5_large_instruct/preprocessed/"
    supervised_only_folder = os.path.join(base_folder, "supervised_only")
    supervised_with_detailed_folder = os.path.join(base_folder, "supervised_with_detailed")
    
    # 폴더 생성
    os.makedirs(supervised_only_folder, exist_ok=True)
    os.makedirs(supervised_with_detailed_folder, exist_ok=True)
    
    # random seed 설정
    random_seed = 42
    np.random.seed(random_seed)
    
    print("\n=== 데이터 분할 시작 ===")
    
    # supervised 데이터만 분할
    print("\n--- supervised_only 데이터 분할 ---")
    print(f"supervised 데이터: {len(supervised_df)} 행")
    
    # train:val:test = 8:1:1 비율로 분할
    train_df, temp_df = train_test_split(supervised_df, test_size=0.2, random_state=random_seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=random_seed)
    
    print(f"supervised_only 분할 결과:")
    print(f"  Train: {len(train_df)} 행")
    print(f"  Val: {len(val_df)} 행")
    print(f"  Test: {len(test_df)} 행")
    
    # supervised_only 파일 저장
    train_df.to_csv(os.path.join(supervised_only_folder, "train.csv"), index=False, encoding='utf-8')
    val_df.to_csv(os.path.join(supervised_only_folder, "val.csv"), index=False, encoding='utf-8')
    test_df.to_csv(os.path.join(supervised_only_folder, "test.csv"), index=False, encoding='utf-8')
    
    # supervised + detailed 데이터 분할
    print("\n--- supervised_with_detailed 데이터 분할 ---")
    
    # detailed 데이터와 supervised 데이터 분리
    # detailed 데이터는 120행이므로 마지막 120행이 detailed 데이터라고 가정
    detailed_df = combined_df.tail(120).copy()
    supervised_df_v2 = combined_df.head(len(combined_df) - 120).copy()
    
    print(f"supervised 데이터: {len(supervised_df_v2)} 행")
    print(f"detailed 데이터: {len(detailed_df)} 행")
    
    # supervised 데이터를 train:val:test = 8:1:1 비율로 분할
    train_supervised, temp_supervised = train_test_split(supervised_df_v2, test_size=0.2, random_state=random_seed)
    val_supervised, test_supervised = train_test_split(temp_supervised, test_size=0.5, random_state=random_seed)
    
    # detailed 데이터를 train에 추가
    train_combined = pd.concat([train_supervised, detailed_df], ignore_index=True)
    
    print(f"supervised_with_detailed 분할 결과:")
    print(f"  Train: {len(train_combined)} 행 (supervised: {len(train_supervised)}, detailed: {len(detailed_df)})")
    print(f"  Val: {len(val_supervised)} 행")
    print(f"  Test: {len(test_supervised)} 행")
    
    # supervised_with_detailed 파일 저장
    train_combined.to_csv(os.path.join(supervised_with_detailed_folder, "train.csv"), index=False, encoding='utf-8')
    val_supervised.to_csv(os.path.join(supervised_with_detailed_folder, "val.csv"), index=False, encoding='utf-8')
    test_supervised.to_csv(os.path.join(supervised_with_detailed_folder, "test.csv"), index=False, encoding='utf-8')
    
    print(f"\n=== 분할 완료 ===")
    print(f"저장 위치:")
    print(f"  supervised_only: {supervised_only_folder}")
    print(f"  supervised_with_detailed: {supervised_with_detailed_folder}")
    print(f"생성된 파일:")
    print(f"  supervised_only: train.csv, val.csv, test.csv")
    print(f"  supervised_with_detailed: train.csv, val.csv, test.csv")

def main():
    """메인 함수"""
    print("=== 데이터 처리 및 분할 시작 ===")
    
    # 1단계: CSV 파일들 합치기
    print("\n1단계: CSV 파일들 합치기")
    supervised_df, combined_df = merge_supervised_csv_files()
    
    if supervised_df is None or combined_df is None:
        print("데이터 합치기에 실패했습니다.")
        return
    
    # 2단계: 데이터 분할
    print("\n2단계: 데이터 분할")
    split_data(supervised_df, combined_df)
    
    print("\n=== 전체 작업 완료 ===")

if __name__ == "__main__":
    main()

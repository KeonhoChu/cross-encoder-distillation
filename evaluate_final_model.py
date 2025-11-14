#!/usr/bin/env python3
"""
최종 Bi-Encoder 모델 평가 스크립트
코사인 유사도 성능 및 Uniformity 분석
"""

import os
import pandas as pd
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse
import json
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def fmt_query(text: str, lang_tag: str = "(ko)") -> str:
    """Query 포맷팅"""
    return f"query: {lang_tag} {text}"

def fmt_passage(text: str, lang_tag: str = "(ko)") -> str:
    """Passage 포맷팅"""
    return f"passage: {lang_tag} {text}"

def normalize_cosine_similarity(cosine_sim: torch.Tensor) -> torch.Tensor:
    """코사인 유사도를 0~1 범위로 정규화 (높은 유사도=1.0, 낮은 유사도=0.0)"""
    return (cosine_sim + 1) / 2

def load_triplets(path: str) -> List[Tuple[str, str, str]]:
    """Triplet 데이터 로드: (anchor, positive, negative)"""
    df = pd.read_csv(path)
    required = {"anchor", "positive", "negative"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"{os.path.basename(path)}에는 {required} 컬럼이 필요합니다.")
    triplets = []
    for _, row in df.iterrows():
        triplets.append((row["anchor"], row["positive"], row["negative"]))
    return triplets

def evaluate_bi_encoder(model_path: str, test_filepath: str, output_dir: str, batch_size: int = 16):
    """Bi-Encoder 코사인 유사도 평가"""
    print(f"[모델 로드] {model_path}")
    
    # SentenceTransformer 모델 로드
    model = SentenceTransformer(model_path)
    model.eval()
    
    print(f"[모델] 로드 완료!")

    print(f"[데이터 로드] {test_filepath}")
    triplets = load_triplets(test_filepath)
    print(f"[데이터] {len(triplets)}개 triplet 로드 완료!")

    print(f"============================================================")
    print(f"Bi-Encoder 코사인 유사도 평가 시작")
    print(f"============================================================")

    all_pos_scores = []
    all_neg_scores = []
    all_margins = []
    
    # 배치 처리로 평가
    for i in tqdm(range(0, len(triplets), batch_size), desc="Evaluating"):
        batch_triplets = triplets[i:i+batch_size]
        
        # 배치 데이터 준비
        anchors = []
        positives = []
        negatives = []
        
        for anchor, positive, negative in batch_triplets:
            anchors.append(fmt_query(anchor))
            positives.append(fmt_passage(positive))
            negatives.append(fmt_passage(negative))
        
        # 임베딩 생성
        with torch.no_grad():
            anchor_embeddings = model.encode(anchors, convert_to_tensor=True)
            positive_embeddings = model.encode(positives, convert_to_tensor=True)
            negative_embeddings = model.encode(negatives, convert_to_tensor=True)
        
        # 코사인 유사도 계산 및 정규화 (0~1 범위)
        pos_similarities_raw = torch.cosine_similarity(anchor_embeddings, positive_embeddings, dim=1)
        neg_similarities_raw = torch.cosine_similarity(anchor_embeddings, negative_embeddings, dim=1)
        
        pos_similarities = normalize_cosine_similarity(pos_similarities_raw)
        neg_similarities = normalize_cosine_similarity(neg_similarities_raw)
        
        # Margin 계산
        margins = pos_similarities - neg_similarities
        
        all_pos_scores.extend(pos_similarities.cpu().numpy())
        all_neg_scores.extend(neg_similarities.cpu().numpy())
        all_margins.extend(margins.cpu().numpy())

    # 통계 계산
    pos_scores = np.array(all_pos_scores)
    neg_scores = np.array(all_neg_scores)
    margins = np.array(all_margins)
    
    pos_mean = np.mean(pos_scores)
    pos_std = np.std(pos_scores)
    neg_mean = np.mean(neg_scores)
    neg_std = np.std(neg_scores)
    margin_mean = np.mean(margins)
    margin_std = np.std(margins)
    
    # Uniformity 계산 (점수 분포의 균일성)
    pos_uniformity = np.var(pos_scores)
    neg_uniformity = np.var(neg_scores)
    overall_uniformity = np.var(np.concatenate([pos_scores, neg_scores]))
    
    print(f"\n============================================================")
    print(f"평가 결과")
    print(f"============================================================")
    print(f"Positive 정규화 점수(0-1): {pos_mean:.4f} ± {pos_std:.4f}")
    print(f"Negative 정규화 점수(0-1): {neg_mean:.4f} ± {neg_std:.4f}")
    print(f"Margin (Pos - Neg): {margin_mean:.4f} ± {margin_std:.4f}")
    print(f"Positive Uniformity (Variance): {pos_uniformity:.4f}")
    print(f"Negative Uniformity (Variance): {neg_uniformity:.4f}")
    print(f"Overall Uniformity (Variance): {overall_uniformity:.4f}")
    print(f"Total samples: {len(triplets)}")
    print(f"============================================================")

    # 결과 저장
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'pos_cosine_similarity_mean': float(pos_mean),
        'pos_cosine_similarity_std': float(pos_std),
        'neg_cosine_similarity_mean': float(neg_mean),
        'neg_cosine_similarity_std': float(neg_std),
        'margin_mean': float(margin_mean),
        'margin_std': float(margin_std),
        'pos_uniformity': float(pos_uniformity),
        'neg_uniformity': float(neg_uniformity),
        'overall_uniformity': float(overall_uniformity),
        'total_samples': len(triplets)
    }
    
    with open(os.path.join(output_dir, 'bi_encoder_evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[저장] 평가 결과: {os.path.join(output_dir, 'bi_encoder_evaluation_results.json')}")

    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 코사인 유사도 분포
    axes[0, 0].hist(pos_scores, bins=50, alpha=0.7, label='Positive 코사인 유사도', color='green')
    axes[0, 0].hist(neg_scores, bins=50, alpha=0.7, label='Negative 코사인 유사도', color='red')
    axes[0, 0].set_title('코사인 유사도 분포')
    axes[0, 0].set_xlabel('코사인 유사도')
    axes[0, 0].set_ylabel('빈도')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Margin 분포
    axes[0, 1].hist(margins, bins=50, alpha=0.7, color='purple')
    axes[0, 1].set_title('Margin 분포 (Positive - Negative)')
    axes[0, 1].set_xlabel('Margin')
    axes[0, 1].set_ylabel('빈도')
    axes[0, 1].grid(True)
    
    # Box plot
    data_to_plot = [pos_scores, neg_scores]
    axes[1, 0].boxplot(data_to_plot, labels=['Positive', 'Negative'])
    axes[1, 0].set_title('코사인 유사도 Box Plot')
    axes[1, 0].set_ylabel('코사인 유사도')
    axes[1, 0].grid(True)
    
    # Uniformity 비교
    uniformity_data = [pos_uniformity, neg_uniformity, overall_uniformity]
    axes[1, 1].bar(['Pos Uniformity', 'Neg Uniformity', 'Overall Uniformity'], uniformity_data, color=['green', 'red', 'blue'])
    axes[1, 1].set_title('Uniformity 비교')
    axes[1, 1].set_ylabel('분산')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'bi_encoder_evaluation_analysis.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"[저장] 분석 그래프: {plot_path}")

    # 샘플 테스트
    print(f"\n============================================================")
    print(f"샘플 테스트")
    print(f"============================================================")
    
    # 첫 번째 triplet으로 테스트
    anchor, positive, negative = triplets[0]
    
    anchor_text = fmt_query(anchor)
    positive_text = fmt_passage(positive)
    negative_text = fmt_passage(negative)
    
    with torch.no_grad():
        anchor_emb = model.encode([anchor_text], convert_to_tensor=True)
        positive_emb = model.encode([positive_text], convert_to_tensor=True)
        negative_emb = model.encode([negative_text], convert_to_tensor=True)
        
        pos_sim_raw = torch.cosine_similarity(anchor_emb, positive_emb, dim=1)
        neg_sim_raw = torch.cosine_similarity(anchor_emb, negative_emb, dim=1)
        
        pos_sim_normalized = normalize_cosine_similarity(pos_sim_raw).item()
        neg_sim_normalized = normalize_cosine_similarity(neg_sim_raw).item()
    
    print(f"Query: {anchor_text}")
    print(f"Positive: {positive_text}")
    print(f"Negative: {negative_text}")
    print(f"Positive 원본 코사인 유사도: {pos_sim_raw.item():.4f}")
    print(f"Positive 정규화 점수(0-1): {pos_sim_normalized:.4f}")
    print(f"Negative 원본 코사인 유사도: {neg_sim_raw.item():.4f}")
    print(f"Negative 정규화 점수(0-1): {neg_sim_normalized:.4f}")
    print(f"Margin: {pos_sim_normalized - neg_sim_normalized:.4f}")

    print(f"\n평가 완료!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bi-Encoder Cosine Similarity Evaluation")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained Bi-Encoder model directory")
    parser.add_argument("--test_data", type=str, required=True,
                        help="Path to the test CSV file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=16, help="Evaluation batch size")
    
    args = parser.parse_args()

    evaluate_bi_encoder(
        model_path=args.model_path,
        test_filepath=args.test_data,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )

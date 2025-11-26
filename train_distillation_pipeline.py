#!/usr/bin/env python3
"""
통합 지식 증류 학습 스크립트
Cross-Encoder Teacher → Bi-Encoder Student 지식 증류
최종 출력: 코사인 유사도
"""

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import List, Tuple
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


def load_triplets(path: str, num_hard_negatives: int = 0, max_samples: int = None) -> List[Tuple[str, List[str]]]:
    """
    Triplet 데이터를 (query, [doc1, doc2, ...]) 형태로 변환
    Hard negative mining으로 더 많은 문서 후보 생성
    
    Args:
        path: 데이터 파일 경로
        num_hard_negatives: 다른 쿼리의 문서를 hard negative로 사용할 개수
        max_samples: 빠른 테스트를 위한 최대 샘플 수
    """
    df = pd.read_csv(path)
    required = {"anchor", "positive", "negative"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"{os.path.basename(path)}에는 {required} 컬럼이 필요합니다.")
    
    # 빠른 테스트를 위한 샘플링
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"[빠른 테스트] 데이터 샘플링: {max_samples}개만 사용")
    
    rows = []
    
    for i, r in df.iterrows():
        query = f"query: (ko) {str(r['anchor'])}"
        positive = f"passage: (ko) {str(r['positive'])}"
        negative = f"passage: (ko) {str(r['negative'])}"
        
        # 기본 문서 (positive 1개, negative 1개)
        docs = [positive, negative]
        
        # Hard negative mining: 다른 쿼리의 문서들을 negative로 추가
        if num_hard_negatives > 0:
            # 현재 쿼리와 다른 행의 문서들을 hard negative로 사용
            other_indices = np.random.choice(
                len(df), 
                size=min(num_hard_negatives, len(df) - 1), 
                replace=False
            )
            
            for idx in other_indices:
                if idx != i:  # 자기 자신 제외
                    other_row = df.iloc[idx]
                    # 다른 쿼리의 positive와 negative를 hard negative로 추가
                    hard_neg_pos = f"passage: (ko) {str(other_row['positive'])}"
                    hard_neg_neg = f"passage: (ko) {str(other_row['negative'])}"
                    docs.extend([hard_neg_pos, hard_neg_neg])
        
        rows.append((query, docs))
    
    return rows


def fmt_query(text: str, lang_tag: str = "(ko)") -> str:
    """Query 포맷팅"""
    return f"query: {lang_tag} {text}".strip()


def fmt_passage(text: str, lang_tag: str = "(ko)") -> str:
    """Passage 포맷팅"""
    return f"passage: {lang_tag} {text}"

def normalize_cosine_similarity(cosine_sim: torch.Tensor) -> torch.Tensor:
    """코사인 유사도를 0~1 범위로 정규화"""
    return (cosine_sim + 1) / 2


class DistillListDataset(Dataset):
    """지식 증류용 데이터셋 (Hard negative mining 지원)"""
    
    def __init__(self, rows, ce_batch_scorer, K=16, use_hard_negative_mining=True):
        self.rows = rows
        self.ce_batch_scorer = ce_batch_scorer
        self.K = K
        self.use_hard_negative_mining = use_hard_negative_mining
        
        # Hard negative mining용: 모든 passage 수집
        if use_hard_negative_mining:
            self.all_passages = []
            for _, docs in rows:
                self.all_passages.extend(docs)
            self.all_passages = list(set(self.all_passages))  # 중복 제거
            print(f"[Hard Negative Mining] 총 {len(self.all_passages)}개의 고유 문서 사용 가능")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        q, docs = self.rows[idx]
        
        # 문서 수가 K보다 적은 경우
        if len(docs) < self.K:
            if self.use_hard_negative_mining and hasattr(self, 'all_passages'):
                # Hard negative mining: 다른 쿼리의 문서를 랜덤하게 추가
                remaining = self.K - len(docs)
                # 현재 쿼리의 문서를 제외한 다른 문서들 중 선택
                candidate_docs = [d for d in self.all_passages if d not in docs]
                if len(candidate_docs) >= remaining:
                    import random
                    selected = random.sample(candidate_docs, remaining)
                    docs.extend(selected)
                else:
                    # 후보가 부족하면 반복 (최후의 수단)
                    docs = docs + docs[:remaining]
            else:
                # Hard negative mining이 없으면 반복
                docs = docs + docs[:self.K - len(docs)]
        else:
            # 문서가 충분하면 상위 K개 사용
            docs = docs[:self.K]
        
        # Teacher soft label
        scores = self.ce_batch_scorer([(q, d) for d in docs])  # (K,)
        return q, docs, scores


@dataclass
class DistillCollator:
    """지식 증류용 Collator"""
    
    be: SentenceTransformer
    max_length: int = 256

    def __call__(self, batch):
        qs, d_lists, ce_scores = zip(*batch)
        B = len(qs)
        K = len(d_lists[0])

        # Tokenize
        q_feat = self.be.tokenize(list(qs), max_length=self.max_length, 
                                   padding=True, truncation=True)
        d_feat = self.be.tokenize([d for docs in d_lists for d in docs], 
                                   max_length=self.max_length, 
                                   padding=True, truncation=True)

        labels = torch.stack(list(ce_scores), dim=0)  # (B, K)
        return [q_feat, d_feat], labels


class ListwiseKLDistillLoss(nn.Module):
    """Listwise KL Distillation Loss (코사인 유사도 양수 범위 변환 개선)"""
    
    def __init__(self, model: SentenceTransformer, tau: float = 0.2, normalize_similarity: bool = True,
                 contrastive_weight: float = 0.3, negative_penalty_weight: float = 0.4):
        super().__init__()
        self.model = model
        self.tau = tau
        self.normalize_similarity = normalize_similarity  # 코사인 유사도를 [0,1]로 정규화할지 여부
        self.contrastive_weight = contrastive_weight  # Contrastive Loss 가중치
        self.negative_penalty_weight = negative_penalty_weight  # Negative 점수 페널티 가중치

    def forward(self, sentence_features, labels_ce=None, **kwargs):
        """
        sentence_features: [q_feat_dict, d_feat_dict]
        labels_ce: (B, K) teacher scores in [0,1]
        """
        # 1) 임베딩 생성
        q_emb = self.model(sentence_features[0])['sentence_embedding']     # (B, H)
        d_emb = self.model(sentence_features[1])['sentence_embedding']     # (B*K, H)

        B, K = labels_ce.shape
        H = q_emb.size(-1)
        d_emb = d_emb.view(B, K, H)

        # 2) 코사인 유사도 계산 (정규화된 벡터)
        qn = F.normalize(q_emb, dim=-1)
        dn = F.normalize(d_emb, dim=-1)
        s_be_raw = torch.einsum("bh,bkh->bk", qn, dn)  # (B, K), 범위: [-1, 1]
        
        # 3) 코사인 유사도를 양수 범위로 변환 [0, 1]
        if self.normalize_similarity:
            # 방법 1: (cos + 1) / 2로 정규화
            s_be = (s_be_raw + 1.0) / 2.0  # [0, 1] 범위
        else:
            # 방법 2: ReLU를 사용하여 음수를 0으로 클리핑
            s_be = F.relu(s_be_raw)  # [0, 1] 범위 (음수는 0으로)
            # 추가로 1로 스케일링 (선택사항)
            # s_be = s_be / (s_be.max() + 1e-8)  # 최대값으로 나눔
        
        # 4) 리스트 단위 softmax 분포
        # Teacher 분포: 이미 [0,1] 범위
        p = F.softmax((labels_ce / self.tau), dim=-1)  # teacher 분포
        
        # Student 분포: 정규화된 유사도를 사용
        q = F.softmax((s_be / self.tau), dim=-1)  # student 분포

        # 5) KL(p || q) 계산
        kl = torch.sum(p * (torch.log(p + 1e-12) - torch.log(q + 1e-12)), dim=-1).mean()
        
        # 6) Contrastive Loss 추가: Positive와 Negative 간의 margin 확보
        # Teacher 점수 기준으로 Positive와 Negative 구분 (임계값: 0.5)
        positive_mask = labels_ce > 0.5  # Positive 문서 마스크 (B, K)
        negative_mask = labels_ce <= 0.5  # Negative 문서 마스크 (B, K)
        
        contrastive_loss = torch.tensor(0.0, device=s_be.device)
        if positive_mask.any() and negative_mask.any():
            # 각 쿼리(배치)별로 Positive와 Negative의 평균 점수 계산
            margin_losses = []
            for b in range(B):
                pos_mask_b = positive_mask[b]  # (K,)
                neg_mask_b = negative_mask[b]  # (K,)
                
                if pos_mask_b.any() and neg_mask_b.any():
                    pos_scores_b = s_be[b][pos_mask_b]  # 해당 쿼리의 Positive 점수들
                    neg_scores_b = s_be[b][neg_mask_b]  # 해당 쿼리의 Negative 점수들
                    
                    pos_mean_b = pos_scores_b.mean()
                    neg_mean_b = neg_scores_b.mean()
                    
                    # Margin이 최소 0.7 이상이 되도록 (더 큰 margin 요구)
                    # Positive는 0.85 이상, Negative는 0.15 이하를 목표로 함
                    target_margin = 0.7
                    margin_loss_b = F.relu(target_margin - (pos_mean_b - neg_mean_b))
                    margin_losses.append(margin_loss_b)
            
            if len(margin_losses) > 0:
                contrastive_loss = torch.stack(margin_losses).mean()
        
        # 7) Negative 점수 페널티: Negative 점수가 낮아지도록 강제
        negative_penalty = torch.tensor(0.0, device=s_be.device)
        if negative_mask.any():
            # 전체 배치에서 Negative 점수들
            neg_scores = s_be[negative_mask]
            # Negative 점수가 0.15 이하가 되도록 (더 낮게 설정)
            target_neg_score = 0.15
            neg_penalty = F.relu(neg_scores.mean() - target_neg_score)
            negative_penalty = neg_penalty
            
            # 추가: 각 쿼리별 Negative 점수도 낮추기
            per_query_penalties = []
            for b in range(B):
                neg_mask_b = negative_mask[b]
                if neg_mask_b.any():
                    neg_scores_b = s_be[b][neg_mask_b]
                    # 각 쿼리의 Negative 평균도 0.15 이하가 되도록
                    neg_penalty_b = F.relu(neg_scores_b.mean() - target_neg_score)
                    per_query_penalties.append(neg_penalty_b)
            
            if len(per_query_penalties) > 0:
                per_query_penalty = torch.stack(per_query_penalties).mean()
                negative_penalty = negative_penalty + 0.5 * per_query_penalty
        
        # 8) 최종 Loss: KL Divergence + Contrastive Loss + Negative Penalty
        total_loss = kl + self.contrastive_weight * contrastive_loss + self.negative_penalty_weight * negative_penalty
        
        return total_loss


def train(args):
    """학습 실행"""
    print("=" * 60)
    print("Cross-Encoder → Bi-Encoder 지식 증류 학습")
    print("=" * 60)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[장치] {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print(f"[장치] {device} (GPU not available)")
    
    # 데이터 로드
    data_dir = f"preprocessed/{args.data_version}"
    train_path = os.path.join(data_dir, "train.csv")
    
    print(f"[데이터 로드] {train_path}")
    # Hard negative mining으로 더 많은 문서 후보 생성
    # K=16이면 기본 2개 + hard negative 14개 정도 필요
    num_hard_negatives = max(0, (args.K - 2) // 2)  # positive/negative 쌍으로 추가
    print(f"[Hard Negative Mining] 쿼리당 {num_hard_negatives}개의 hard negative 쌍 추가")
    train_rows = load_triplets(train_path, num_hard_negatives=num_hard_negatives, max_samples=args.max_samples)
    print(f"[데이터] {len(train_rows)}개의 쿼리 로드")
    
    # 각 쿼리의 평균 문서 수 확인
    avg_docs = np.mean([len(docs) for _, docs in train_rows])
    print(f"[데이터 통계] 쿼리당 평균 문서 수: {avg_docs:.1f}개")
    
    # Teacher: Cross-Encoder
    print(f"\n[Teacher] Cross-Encoder 로드: {args.teacher_path}")
    from train_teacher_model import CrossEncoderModel
    from transformers import AutoTokenizer, AutoModel
    
    # Teacher 모델의 LoRA 설정 로드
    teacher_config_path = os.path.join(args.teacher_path, 'lora_config.json')
    if os.path.exists(teacher_config_path):
        with open(teacher_config_path, 'r') as f:
            teacher_config = json.load(f)
        teacher_lora_r = teacher_config.get('lora_r', 32)
        teacher_lora_alpha = teacher_config.get('lora_alpha', 64)
        teacher_lora_dropout = teacher_config.get('lora_dropout', 0.1)
        teacher_use_lora = teacher_config.get('use_lora', True)
        print(f"[Teacher LoRA 설정] r={teacher_lora_r}, alpha={teacher_lora_alpha}, dropout={teacher_lora_dropout}")
    else:
        # 설정 파일이 없으면 기본값 사용 (하위 호환성)
        print(f"[경고] Teacher LoRA 설정 파일을 찾을 수 없습니다. 기본값 사용: r=32, alpha=64")
        teacher_lora_r = 32
        teacher_lora_alpha = 64
        teacher_lora_dropout = 0.1
        teacher_use_lora = True
    
    ce = CrossEncoderModel(
        model_name="intfloat/multilingual-e5-large-instruct",
        use_lora=teacher_use_lora,
        lora_r=teacher_lora_r,
        lora_alpha=teacher_lora_alpha,
        lora_dropout=teacher_lora_dropout
    )
    ce.load_state_dict(torch.load(os.path.join(args.teacher_path, "best_model.pt"), map_location="cpu"))
    ce.to(device)
    ce.eval()
    ce_tok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large-instruct")
    
    # GPU 메모리 정리
    torch.cuda.empty_cache()
    
    @torch.no_grad()
    def ce_score_batch(pairs: List[Tuple[str, str]]) -> torch.Tensor:
        """Cross-Encoder 점수 계산"""
        inputs = ce_tok(
            [p[0] for p in pairs], [p[1] for p in pairs],
            padding=True, truncation=True, 
            max_length=args.max_len, return_tensors="pt"
        ).to(device)
        logits = ce(inputs['input_ids'], inputs['attention_mask'])
        probs = torch.sigmoid(logits['logits']).squeeze()
        if probs.dim() == 0:
            probs = probs.unsqueeze(0)
        # GPU 메모리 절약: 즉시 CPU로 이동하고 캐시 정리
        result = probs.detach().cpu()
        del probs, logits, inputs
        torch.cuda.empty_cache()
        return result
    
    # Student: Bi-Encoder
    print(f"\n[Student] Bi-Encoder 로드: {args.student_name}")
    be = SentenceTransformer(args.student_name, device=device)
    
    # 데이터셋 및 로더
    dataset = DistillListDataset(
        train_rows, 
        ce_score_batch, 
        K=args.K,
        use_hard_negative_mining=args.use_hard_negative_mining
    )
    collate = DistillCollator(be, max_length=args.max_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    
    # Loss 및 Optimizer
    loss_fn = ListwiseKLDistillLoss(
        be, 
        tau=args.tau,
        normalize_similarity=args.normalize_similarity,
        contrastive_weight=args.contrastive_weight,
        negative_penalty_weight=args.negative_penalty_weight
    )
    opt = torch.optim.AdamW(be.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 학습 루프
    be.train()
    for epoch in range(args.epochs):
        print(f"\n[Epoch {epoch+1}/{args.epochs}]")
        total_loss = 0.0
        
        for batch_idx, (sentence_features, labels_ce) in enumerate(tqdm(loader, desc="Training")):
            try:
                # to device
                for d in sentence_features:
                    for k, v in d.items():
                        d[k] = v.to(device)
                labels_ce = labels_ce.to(device)
                
                loss = loss_fn(sentence_features, labels_ce)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(be.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                total_loss += float(loss.item())
                
                # 주기적으로 GPU 캐시 정리 (메모리 누수 방지)
                if (batch_idx + 1) % 10 == 0:
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e) or "NVML" in str(e):
                    print(f"\n[경고] GPU 메모리 에러 발생 (배치 {batch_idx+1})")
                    print(f"[조치] GPU 캐시 정리 중...")
                    torch.cuda.empty_cache()
                    opt.zero_grad()
                    # 다음 배치로 건너뛰기
                    continue
                else:
                    raise
        
        avg_loss = total_loss / len(loader)
        print(f"[Train] Loss: {avg_loss:.4f}")
    
    # 모델 저장
    os.makedirs(args.output_dir, exist_ok=True)
    be.save(args.output_dir)
    print(f"\n[저장] 모델 저장: {args.output_dir}")
    
    # 테스트
    print("\n[테스트] 코사인 유사도 출력:")
    be.eval()
    q = train_rows[0][0]
    d = train_rows[0][1][0]
    
    q_emb = be.encode(q, normalize_embeddings=True)
    d_emb = be.encode(d, normalize_embeddings=True)
    
    cos = float(cos_sim(q_emb, d_emb))
    score01 = (cos + 1.0) / 2.0
    print(f"Query: {q}")
    print(f"Passage: {d}")
    print(f"코사인 유사도: {cos:.4f}")
    print(f"정규화 점수(0-1): {score01:.4f}")
    
    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Cross-Encoder → Bi-Encoder 지식 증류')
    parser.add_argument('--teacher_path', type=str, required=True,
                       help='Cross-Encoder 모델 경로')
    parser.add_argument('--student_name', type=str, default='intfloat/multilingual-e5-large-instruct',
                       help='Bi-Encoder 모델 이름')
    parser.add_argument('--data_version', type=str, default='supervised_only',
                       choices=['supervised_only', 'supervised_with_detailed'])
    parser.add_argument('--output_dir', type=str, required=True,
                       help='저장할 모델 경로')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--K', type=int, default=16,
                       help='쿼리당 후보 문서 수 (기본값: 16)')
    parser.add_argument('--tau', type=float, default=0.2,
                       help='Temperature (0.1~0.5 권장, 기본값: 0.2)')
    parser.add_argument('--use_hard_negative_mining', type=bool, default=True,
                       help='Hard negative mining 사용 여부 (기본값: True)')
    parser.add_argument('--normalize_similarity', type=bool, default=True,
                       help='코사인 유사도를 [0,1] 범위로 정규화 (기본값: True)')
    parser.add_argument('--contrastive_weight', type=float, default=0.3,
                       help='Contrastive Loss 가중치 (기본값: 0.3)')
    parser.add_argument('--negative_penalty_weight', type=float, default=0.4,
                       help='Negative 점수 페널티 가중치 (기본값: 0.4, 더 강하게 적용)')
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_lora', action='store_true', default=True)
    parser.add_argument('--lora_r', type=int, default=32)
    parser.add_argument('--lora_alpha', type=int, default=64)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--max_samples', type=int, default=None,
                       help='빠른 테스트를 위한 최대 샘플 수 (예: 1000)')
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

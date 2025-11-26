#!/usr/bin/env python3
"""
1단계: Cross-Encoder 학습 스크립트 (Teacher 모델)
"""

import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse
import json
from typing import List, Tuple, Dict
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType

def fmt_query(text: str, lang_tag: str = "(ko)") -> str:
    """Query 포맷팅"""
    return f"query: {lang_tag} {text}"

def fmt_passage(text: str, lang_tag: str = "(ko)") -> str:
    """Passage 포맷팅"""
    return f"passage: {lang_tag} {text}"

def load_triplets(path: str, max_samples: int = None) -> List[Tuple[str, str, str]]:
    """Triplet 데이터 로드: (anchor, positive, negative)"""
    df = pd.read_csv(path)
    required = {"anchor", "positive", "negative"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"{os.path.basename(path)}에는 {required} 컬럼이 필요합니다.")
    
    # 빠른 테스트를 위한 샘플링
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"[빠른 테스트] 데이터 샘플링: {max_samples}개만 사용")
    
    triplets = []
    for _, row in df.iterrows():
        triplets.append((row["anchor"], row["positive"], row["negative"]))
    return triplets

class CrossEncoderDataset(Dataset):
    """Cross-Encoder 데이터셋"""
    def __init__(self, triplets: List[Tuple[str, str, str]], tokenizer, max_length: int = 512):
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.triplets) * 2  # positive + negative
    
    def __getitem__(self, idx):
        triplet_idx = idx // 2
        is_positive = idx % 2 == 0
        
        anchor, positive, negative = self.triplets[triplet_idx]
        
        if is_positive:
            text1 = fmt_query(anchor)
            text2 = fmt_passage(positive)
            label = 1
        else:
            text1 = fmt_query(anchor)
            text2 = fmt_passage(negative)
            label = 0
        
        # 텍스트 결합
        combined_text = f"{text1} [SEP] {text2}"
        
        # 토크나이징
        encoding = self.tokenizer(
            combined_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

class CrossEncoderModel(nn.Module):
    """Cross-Encoder 모델"""
    def __init__(self, model_name: str, use_lora: bool = False, lora_r: int = 32, 
                 lora_alpha: int = 64, lora_dropout: float = 0.1):
        super().__init__()
        
        # 베이스 모델 로드
        self.base_model = AutoModel.from_pretrained(model_name)
        
        # LoRA 적용
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["query", "key", "value", "dense", "intermediate.dense", "output.dense", "pooler.dense"]
            )
            self.base_model = get_peft_model(self.base_model, lora_config)
        
        # 분류 헤드
        hidden_size = self.base_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask, labels=None):
        # 베이스 모델 출력
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # [CLS] 토큰 사용
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        
        # 분류
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits.squeeze(), labels)
        
        return {
            'logits': logits,
            'loss': loss
        }

def train(model_name: str, data_version: str, output_dir: str, epochs: int = 3, 
          batch_size: int = 16, lr: float = 2e-5, use_lora: bool = True,
          lora_r: int = 32, lora_alpha: int = 64, lora_dropout: float = 0.1,
          max_samples: int = None):
    """Cross-Encoder 학습"""
    
    print(f"============================================================")
    print(f"1단계: Cross-Encoder Teacher 학습 시작")
    print(f"============================================================")
    print(f"모델: {model_name}")
    print(f"데이터: {data_version}")
    print(f"출력: {output_dir}")
    print(f"에포크: {epochs}, 배치크기: {batch_size}, 학습률: {lr}")
    print(f"LoRA: {use_lora} (r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout})")
    print(f"============================================================")
    
    # 데이터 로드
    base_data_path = "preprocessed"
    train_path = os.path.join(base_data_path, data_version, "train.csv")
    val_path = os.path.join(base_data_path, data_version, "val.csv")
    
    print(f"[데이터 로드] {train_path}")
    train_triplets = load_triplets(train_path, max_samples=max_samples)
    print(f"[훈련 데이터] {len(train_triplets)}개 triplet")
    
    print(f"[데이터 로드] {val_path}")
    val_triplets = load_triplets(val_path, max_samples=None)  # 검증은 전체 사용
    print(f"[검증 데이터] {len(val_triplets)}개 triplet")
    
    # 장치 설정
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[장치] {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print(f"[장치] {device} (GPU not available)")
    
    # 토크나이저 및 모델 초기화
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = CrossEncoderModel(model_name, use_lora, lora_r, lora_alpha, lora_dropout)
    model = model.to(device)  # GPU로 이동
    
    # 데이터셋 및 데이터로더
    train_dataset = CrossEncoderDataset(train_triplets, tokenizer)
    val_dataset = CrossEncoderDataset(val_triplets, tokenizer)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 옵티마이저
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # 학습 루프
    model.train()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\n[에포크 {epoch+1}/{epochs}]")
        
        # 훈련
        epoch_loss = 0
        num_batches = 0
        
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()
            
            # GPU로 데이터 이동
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = epoch_loss / num_batches
        print(f"[훈련] 평균 Loss: {avg_train_loss:.4f}")
        
        # 검증
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}"):
                # GPU로 데이터 이동
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs['loss'].item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        print(f"[검증] 평균 Loss: {avg_val_loss:.4f}")
        
        # 최고 모델 저장
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
            # LoRA 설정 저장
            config = {
                'use_lora': use_lora,
                'lora_r': lora_r,
                'lora_alpha': lora_alpha,
                'lora_dropout': lora_dropout
            }
            with open(os.path.join(output_dir, 'lora_config.json'), 'w') as f:
                json.dump(config, f, indent=2)
            print(f"[저장] 최고 모델 저장: {output_dir}")
        
        model.train()
    
    print(f"\n============================================================")
    print(f"1단계 Cross-Encoder Teacher 학습 완료!")
    print(f"최종 Loss: {best_loss:.4f}")
    print(f"모델 저장: {output_dir}")
    print(f"============================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1단계: Cross-Encoder Teacher 학습")
    parser.add_argument("--model_name", type=str, default="intfloat/multilingual-e5-large-instruct",
                        help="사전 훈련된 모델 이름")
    parser.add_argument("--data_version", type=str, default="supervised_only",
                        choices=['supervised_only', 'supervised_with_detailed'],
                        help="데이터 버전")
    parser.add_argument("--output_dir", type=str, default="./models/cross_encoder_teacher",
                        help="출력 디렉토리")
    parser.add_argument("--epochs", type=int, default=3, help="학습 에포크 수")
    parser.add_argument("--batch_size", type=int, default=16, help="배치 크기")
    parser.add_argument("--lr", type=float, default=2e-5, help="학습률")
    parser.add_argument("--use_lora", action="store_true", help="LoRA 사용")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--max_samples", type=int, default=None, 
                        help="빠른 테스트를 위한 최대 샘플 수 (예: 1000)")
    
    args = parser.parse_args()
    
    train(
        model_name=args.model_name,
        data_version=args.data_version,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_samples=args.max_samples
    )
import pandas as pd
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. 일상 대화 데이터 로드 (실습 데이터 기반 샘플링)
dataset = load_dataset("smilegate-ai/kor_unsmile")
df = pd.DataFrame(dataset["train"]).head(5000) 
df = df.rename(columns={'문장': 'dialogue'}) 

# 2. 모델 로드 및 임베딩
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
dialogue_embeddings = model.encode(df['dialogue'].tolist(), show_progress_bar=True, convert_to_numpy=True)

# 3. 유사 대화 검색 함수
def search_similar_dialogue(user_query: str, top_k: int = 3):
    query_emb = model.encode([user_query], convert_to_numpy=True)
    similarities = cosine_similarity(query_emb, dialogue_embeddings)[0]
    top_indices = np.argsort(-similarities)[:top_k]
    
    print(f"\n🗣️ 사용자 입력: '{user_query}'")
    print("🔎 가장 유사한 대화 문맥 검색 결과:")
    for rank, idx in enumerate(top_indices, 1):
        sim_score = similarities[idx]
        matched_text = df.iloc[idx]['dialogue']
        print(f"[{rank}위 | 유사도 {sim_score:.3f}] {matched_text}")

# 4. 테스트 실행
test_queries = [
    "오늘 날씨가 너무 좋아서 산책 가고 싶어.",
    "요즘 회사 일이 너무 많아서 스트레스 받아."
]

for q in test_queries:
    search_similar_dialogue(q)

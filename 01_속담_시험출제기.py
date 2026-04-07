import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. 속담 데이터 구축 (실습 데이터 기반 샘플)
proverbs_data = [
    {"proverb": "가는 말이 고와야 오는 말이 곱다", "meaning": "자기가 남에게 말이나 행동을 좋게 해야 남도 자기에게 좋게 한다는 말"},
    {"proverb": "개구리 올챙이 적 생각 못 한다", "meaning": "형편이 나아진 뒤에 지난날의 미천하거나 어려웠던 때의 일을 생각하지 못하고 처음부터 잘난 듯이 뽐냄"},
    {"proverb": "소 잃고 외양간 고친다", "meaning": "이미 일을 그르친 뒤에는 뉘우쳐도 소용이 없다는 말"},
    {"proverb": "원숭이도 나무에서 떨어진다", "meaning": "아무리 익숙하고 잘하는 사람이라도 실수할 때가 있다는 말"}
]
df = pd.DataFrame(proverbs_data)

# 2. 문장 임베딩 모델 로드 및 데이터 임베딩
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
proverb_embeddings = model.encode(df['meaning'].tolist(), convert_to_numpy=True)

# 3. 코사인 유사도 기반 검색 함수
def find_best_proverb(situation_text: str, top_k: int = 2):
    query_embedding = model.encode([situation_text], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, proverb_embeddings)[0]
    top_indices = np.argsort(-similarities)[:top_k]
    
    print("\n=======================================")
    print(f"📝 입력된 상황: {situation_text}")
    print("💡 추천 속담 (Top-k):")
    for idx in top_indices:
        print(f"- {df.iloc[idx]['proverb']} (유사도: {similarities[idx]:.3f})")
    print("=======================================\n")

# 4. 테스트 실행
situations = [
    "친구가 맨날 자기는 게임 엄청 잘한다고 자랑하더니, 막상 대회 나가서 예선 탈락했어.",
    "내가 먼저 친구한테 친절하게 대하니까, 친구도 나한테 엄청 잘해주네."
]

for sit in situations:
    find_best_proverb(sit)

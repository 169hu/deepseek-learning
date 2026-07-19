import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'   # 设置镜像

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 使用你本机已有的模型缓存，如果还没有会从镜像下载
model = SentenceTransformer('all-MiniLM-L6-v2')

# 后续代码不变...-L12-v2')

# 将文字转换为向量（一串数字）
text1 = "苹果"
text2 = "香蕉"
text3 = "宇宙飞船"

emb1 = model.encode(text1)
emb2 = model.encode(text2)
emb3 = model.encode(text3)

# 计算余弦相似度（数值越接近1表示越相关）
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

sim_apple_banana = cosine_similarity([emb1], [emb2])[0][0]
sim_apple_spaceship = cosine_similarity([emb1], [emb3])[0][0]

print(f"苹果 vs 香蕉 相似度：{sim_apple_banana:.4f}")   # 应该 > 0.6
print(f"苹果 vs 宇宙飞船 相似度：{sim_apple_spaceship:.4f}")  # 应该 < 0.3
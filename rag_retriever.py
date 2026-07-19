import os
# 强制离线模式，避免网络超时
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import chromadb
from sentence_transformers import SentenceTransformer

# ==================== 1. 加载 Embedding 模型 ====================
print("正在加载模型，请稍候...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("模型加载完成！")

# ==================== 2. 读取并切分文档 ====================
with open('data/my_doc.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 按句号切分
chunks = []
for sentence in text.split('。'):
    if sentence.strip():
        chunks.append(sentence.strip() + '。')

print(f"文档已切分为 {len(chunks)} 个片段")

# ==================== 3. 向量化 ====================
print("正在将文档片段转换为向量...")
chunk_embeddings = model.encode(chunks)
print("向量转换完成！")

# ==================== 4. 存入 ChromaDB ====================
print("正在存入向量数据库...")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="my_docs")

# 如果集合里已有数据，先清空
if collection.count() > 0:
    print("检测到旧数据，正在清理...")
    # 直接获取所有 ID 列表并删除
    all_ids = collection.get()['ids']
    if all_ids:
        collection.delete(ids=all_ids)

# 准备新数据
ids = [f"id_{i}" for i in range(len(chunks))]

collection.add(
    embeddings=chunk_embeddings.tolist(),
    documents=chunks,
    ids=ids
)
print(f"成功存入 {collection.count()} 个文档片段！")

# ==================== 5. 测试检索 ====================
print("\n" + "="*30)
print("开始测试检索功能")

query = "什么是 RAG？"
print(f"你的问题：{query}")

query_embedding = model.encode([query])
results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=2
)

print("\n找到的最相关片段如下：")
for i, (doc, dist) in enumerate(zip(results['documents'][0], results['distances'][0])):
    print(f"\n片段 {i+1}（距离值：{dist:.4f}）：")
    print(f"  {doc}")
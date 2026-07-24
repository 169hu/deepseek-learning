import os
# ==== 强制离线（必须在所有 import 之前） ====
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 其余代码保持不变...

# ==================== 1. 加载 Embedding 模型 ====================
print("正在加载模型，请稍候...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("模型加载完成！")

# ==================== 2. 读取文档 ====================
with open('data/my_doc.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"原始文档长度：{len(text)} 字符")

# ==================== 3. 父子文档切分 ====================
# 父文档：按段落/大块切分（500 字符左右）
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "，", " ", ""],
    length_function=len,
)

# 子文档：按小块切分（100 字符左右，用于检索）
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separators=["。", "！", "？", "\n", "，", " ", ""],
    length_function=len,
)

# 先切父文档
parent_chunks = parent_splitter.split_text(text)
print(f"生成 {len(parent_chunks)} 个父文档（大块）")

# 再对每个父文档切子文档
all_child_chunks = []
parent_map = []  # 记录每个子文档对应的父文档索引和内容

for idx, parent in enumerate(parent_chunks):
    children = child_splitter.split_text(parent)
    for child in children:
        all_child_chunks.append(child)
        parent_map.append({
            "child": child,
            "parent_idx": idx,
            "parent_content": parent
        })

print(f"生成 {len(all_child_chunks)} 个子文档（小块，用于检索）")

# ==================== 4. 向量化子文档 ====================
print("正在将子文档转换为向量...")
child_embeddings = model.encode(all_child_chunks)
print("向量转换完成！")

# ==================== 5. 存入 ChromaDB ====================
print("正在存入向量数据库...")
client = chromadb.PersistentClient(path="./chroma_db")

# 删除旧 collection（如果存在）
try:
    client.delete_collection("my_docs_parent")
except:
    pass

collection = client.create_collection(name="my_docs_parent")

# 准备数据
ids = [f"child_{i}" for i in range(len(all_child_chunks))]
metadatas = [
    {
        "parent_idx": item["parent_idx"],
        "parent_content": item["parent_content"][:500]  # 截断存储，避免超长
    }
    for item in parent_map
]

# 分批插入（避免数据量过大）
batch_size = 100
for i in range(0, len(all_child_chunks), batch_size):
    batch_end = min(i + batch_size, len(all_child_chunks))
    collection.add(
        embeddings=child_embeddings[i:batch_end].tolist(),
        documents=all_child_chunks[i:batch_end],
        ids=ids[i:batch_end],
        metadatas=metadatas[i:batch_end]
    )
    print(f"已存入 {batch_end}/{len(all_child_chunks)} 个片段")

print(f"✅ 成功存入 {collection.count()} 个子文档片段！")
print(f"父文档数量：{len(parent_chunks)}")
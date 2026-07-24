import os
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# ==================== 配置 ====================
load_dotenv()
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 加载模型和向量库
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="my_docs_parent")

# ==================== 定义测试问题 ====================
test_questions = [
    "什么是 RAG？",
    "DeepSeek 是什么？",
    "DeepSeek 有哪些功能？",
    "什么是 ChromaDB？",
    "DeepSeek 的 API 兼容什么格式？",
    "RAG 如何减少模型幻觉？",
    "向量数据库在 RAG 中起什么作用？",
    "DeepSeek 能进行哪些自然语言处理任务？"
]


# ==================== 定义 RAG 检索函数 ====================
def rag_retrieve(question):
    """检索并生成回答"""
    # 检索
    query_embedding = model.encode([question])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3
    )

    retrieved_children = results['documents'][0]
    retrieved_metadatas = results['metadatas'][0]

    # 提取父文档
    parent_docs = []
    seen = set()
    for meta in retrieved_metadatas:
        parent_content = meta.get('parent_content', '')
        if parent_content and parent_content not in seen:
            parent_docs.append(parent_content)
            seen.add(parent_content)
            if len(parent_docs) >= 3:
                break

    if len(parent_docs) < 3:
        for child in retrieved_children:
            if child not in seen:
                parent_docs.append(child)
                seen.add(child)
                if len(parent_docs) >= 3:
                    break

    top_docs = parent_docs[:3]
    context = "\n---\n".join(top_docs)

    # 生成回答
    system_prompt = (
        "你是一个知识库问答助手。请根据【参考内容】回答用户的问题。\n"
        "规则：\n"
        "1. 如果参考内容中有相关信息，请基于这些信息给出准确、简洁的回答。\n"
        "2. 如果参考内容中没有相关信息，请明确回答「根据现有资料无法回答」。\n"
        "3. 不要编造参考内容之外的信息。"
    )
    user_prompt = f"【参考内容】\n{context}\n\n【用户问题】\n{question}"

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )

    return {
        "answer": response.choices[0].message.content,
        "contexts": top_docs
    }


# ==================== 运行评估 ====================
print("🔄 正在为每个问题生成回答...")
answers = []
contexts = []

for q in test_questions:
    print(f"  处理问题：{q[:30]}...")
    result = rag_retrieve(q)
    answers.append(result["answer"])
    contexts.append(result["contexts"])

# 准备评估数据集
data = {
    "question": test_questions,
    "answer": answers,
    "contexts": contexts
}
dataset = Dataset.from_dict(data)

print("\n📊 正在运行 RAGAS 评估...")
result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])

print("\n" + "=" * 40)
print("📊 评估结果")
print("=" * 40)
print(f"Faithfulness（忠实度）: {result['faithfulness']:.4f}")
print(f"Answer Relevancy（答案相关性）: {result['answer_relevancy']:.4f}")
print("=" * 40)

if result['faithfulness'] > 0.7 and result['answer_relevancy'] > 0.7:
    print("✅ 两个指标均超过 0.7，RAG 系统质量良好！")
else:
    print("⚠️ 部分指标偏低，建议检查文档质量和检索策略。")
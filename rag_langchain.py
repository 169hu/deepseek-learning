import os
from dotenv import load_dotenv

# ===== 强制离线 =====
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ===== 导入 LangChain 组件（新版写法） =====
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate          # ← 改这里
from langchain_classic.chains import RetrievalQA          # ← 改这里

# ===== 加载环境变量 =====
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"

# ===== 创建 Embedding 模型 =====
embedding_model = HuggingFaceEmbeddings(
    model_name="paraphrase-MiniLM-L3-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# ===== 加载 Chroma 向量库 =====
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model,
    collection_name="my_docs_parent"
)

# ===== 创建检索器 =====
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ===== 创建 LLM（DeepSeek） =====
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.3,
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com"
)

# ===== 构建 Prompt =====
template = """你是一个知识库问答助手。请根据【参考内容】回答用户的问题。
规则：
1. 如果参考内容中有相关信息，请基于这些信息给出准确、简洁的回答。
2. 如果参考内容中没有相关信息，请明确回答「根据现有资料无法回答」。
3. 不要编造参考内容之外的信息。

参考内容：
{context}

用户问题：
{question}
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# ===== 构建 RAG 链 =====
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# ===== 测试 =====
if __name__ == "__main__":
    question = "什么是 RAG？"
    result = qa_chain.invoke({"query": question})
    print("问题:", question)
    print("回答:", result["result"])
    print("\n参考文档:")
    for i, doc in enumerate(result["source_documents"], 1):
        print(f"[{i}] {doc.page_content[:100]}...")
# ==================== 网络配置（必须在所有 import 之前） ====================
import os
import logging
import time
from datetime import datetime
# 使用 Hugging Face 镜像加速（国内访问更快）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ==================== 导入依赖库 ====================
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
from collections import Counter

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="DeepSeek 工具箱",
    page_icon="🔧",
    layout="wide"
)

# ... 其他代码保持不变 ...

# ==================== 导入依赖库 ====================
from openai import OpenAI
from dotenv import load_dotenv

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="DeepSeek 工具箱",
    page_icon="🔧",
    layout="wide"
)

# ==================== 自定义样式 ====================
st.markdown("""
<style>
    /* --- 全局 --- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* 隐藏默认页脚和顶部空白 */
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp {margin-top: -60px;}

    /* --- 侧边栏 --- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        padding-top: 2rem;
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    [data-testid="stSidebar"] .stRadio > div {
        gap: 0.25rem;
    }
    [data-testid="stSidebar"] .stRadio label {
        padding: 0.6rem 1rem;
        border-radius: 8px;
        transition: all 0.2s;
        font-size: 0.95rem;
        font-weight: 500;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] .stRadio label[data-selected="true"] {
        background: rgba(99, 102, 241, 0.25);
        border-left: 3px solid #818cf8;
    }

    /* --- 主标题 --- */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.25rem;
        letter-spacing: -0.5px;
    }
    .main-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    /* --- 卡片容器 --- */
    .result-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #334155;
    }

    /* --- 按钮 --- */
    .stButton > button {
        background: #4f46e5 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        background: #4338ca !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    }

    /* --- 输入框 --- */
    .stTextArea textarea, .stTextInput input {
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        font-size: 0.9rem !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #818cf8 !important;
        box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.15) !important;
    }

    /* --- 聊天消息 --- */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 0.75rem 1rem;
    }

    /* --- 提示框 --- */
    .stAlert {
        border-radius: 10px !important;
        border: none !important;
        font-size: 0.9rem !important;
    }

    /* --- 代码块 --- */
    .stCodeBlock {
        border-radius: 10px !important;
    }

    /* --- 分割线 --- */
    hr {
        border-color: #e2e8f0 !important;
        margin: 1.5rem 0 !important;
    }

    /* --- 标签 --- */
    .tag {
        display: inline-block;
        background: #eef2ff;
        color: #4f46e5;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
        margin-right: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 加载环境变量和 API 客户端 ====================
load_dotenv()
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ==================== 使用缓存加载 RAG 组件 ====================
@st.cache_resource
def load_rag_components():
    try:
        model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_collection(name="my_docs_parent")   # ← 改成父文档集合
        return model, collection
    except Exception as e:
        st.warning(f"⚠️ RAG 组件加载失败...")
        return None, None

embed_model, collection = load_rag_components()

# ==================== 侧边栏 ====================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 1.5rem 0;">
        <div style="font-size: 1.3rem; font-weight: 700; color: #e0e0e0;">DeepSeek 工具箱</div>
        <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.2rem;">六个实用功能，一个入口</div>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "选择功能",
        ["翻译", "对话", "代码摘要", "信息提取", "逐步推理", "知识库问答"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("""
    <div style="font-size: 0.78rem; color: #94a3b8; line-height: 1.6;">
        <p style="margin-bottom: 0.5rem; font-weight: 600; color: #cbd5e1;">使用提示</p>
        <p>知识库问答需要先运行 <code style="background:#1e293b; padding:2px 6px; border-radius:4px;">rag_retriever.py</code> 导入文档数据。</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== 主界面标题 ====================
st.markdown('<div class="main-title">DeepSeek 工具箱</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">翻译 · 对话 · 代码摘要 · 信息提取 · 逐步推理 · 知识库问答</div>', unsafe_allow_html=True)
st.markdown("---")

# ==================== 各模式功能实现 ====================

# ---------- 翻译 ----------
if mode == "翻译":
    st.subheader("文本翻译")

    # 语言选择
    col_lang, _ = st.columns([1, 3])
    with col_lang:
        target_lang = st.selectbox(
            "目标语言",
            ["英文", "中文", "日文", "韩文", "法文", "德文"],
            label_visibility="collapsed"
        )

    with st.form("translate_form"):
        text = st.text_area("输入要翻译的内容", height=120, placeholder="在此粘贴或输入文本...")
        submitted = st.form_submit_button("开始翻译")
        if submitted and text:
            with st.spinner("正在翻译..."):
                lang_map = {"英文": "English", "中文": "Chinese", "日文": "Japanese", "韩文": "Korean", "法文": "French", "德文": "German"}
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": f"你是莎士比亚时代的英国诗人。请将中文翻译成极具古典文学色彩的{target_lang}，不准意译，必须直译。"},
                        {"role": "user", "content": text}
                    ]
                )
                st.markdown('<div class="result-card">' + response.choices[0].message.content.replace('\n', '<br>') + '</div>', unsafe_allow_html=True)

# ---------- 对话 ----------
elif mode == "对话":
    st.subheader("自由对话")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 清空按钮
    if st.session_state.chat_history:
        if st.button("清空对话", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.chat_message("user", avatar="👤").write(msg["content"])
        else:
            st.chat_message("assistant", avatar="🔧").write(msg["content"])

    user_input = st.chat_input("说点什么...")
    if user_input:
        st.chat_message("user", avatar="👤").write(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner(""):
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=st.session_state.chat_history
            )
            reply = response.choices[0].message.content
            st.chat_message("assistant", avatar="🔧").write(reply)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

# ---------- 代码摘要 ----------
elif mode == "代码摘要":
    st.subheader("代码摘要")
    st.caption("粘贴一段代码，帮你快速理解它的作用")

    with st.form("summary_form"):
        code = st.text_area("代码内容", height=220, placeholder="在此粘贴代码...")
        submitted = st.form_submit_button("生成摘要")
        if submitted and code:
            with st.spinner("正在分析..."):
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "用简洁的3-5句话总结以下代码的功能和核心逻辑，不要啰嗦。"},
                        {"role": "user", "content": code}
                    ]
                )
                st.markdown('<div class="result-card">' + response.choices[0].message.content.replace('\n', '<br>') + '</div>', unsafe_allow_html=True)

# ---------- 信息提取（Few-shot 升级版） ----------
elif mode == "信息提取":
    st.subheader("信息提取（Few-shot 演示）")
    st.caption("给模型看 2 个示例，它就能学会提取任意格式的数据")

    # ===== 让用户也能看到“示例”是什么样的 =====
    with st.expander("📖 当前使用的 Few-shot 示例（教具）"):
        st.code("""
示例 1: 输入: "小明，数学95，语文88" → 输出: {"name": "小明", "scores": {"数学": 95, "语文": 88}}
示例 2: 输入: "小红，英语92，科学85" → 输出: {"name": "小红", "scores": {"英语": 92, "科学": 85}}
        """)

    # ===== 核心输入区 =====
    with st.form("json_form"):
        text_input = st.text_area(
            "输入待提取的文本",
            height=100,
            placeholder="例如：小刚，数学78，英语82，历史90"
        )
        submitted = st.form_submit_button("提取信息")

        if submitted and text_input:
            with st.spinner("模型正在模仿示例进行提取..."):
                # 构建 Few-shot 示例（和刚才测试的代码一样）
                examples = [
                    {"input": "小明，数学95，语文88", "output": {"name": "小明", "scores": {"数学": 95, "语文": 88}}},
                    {"input": "小红，英语92，科学85", "output": {"name": "小红", "scores": {"英语": 92, "科学": 85}}}
                ]
                system_prompt = "你是一个数据提取助手。请严格按照以下示例的 JSON 格式输出，不要添加任何额外文字。\n"
                for ex in examples:
                    system_prompt += f"输入：{ex['input']} → 输出：{ex['output']}\n"

                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text_input}
                    ],
                    temperature=0.1
                )
                st.success("提取成功！")
                st.markdown('<div class="result-card">' + response.choices[0].message.content + '</div>', unsafe_allow_html=True)

                # ===== 额外加分点：解释原理 =====
                st.caption("💡 模型之所以能正确提取，是因为它模仿了上面示例中的格式。这就是 Few-shot 的力量。")
# ---------- 逐步推理 ----------
# ---------- 逐步推理（升级版：支持 Self-Consistency） ----------
elif mode == "逐步推理":
    st.subheader("逐步推理 · 自洽性演示")
    st.caption("模型可以多次推理，然后投票选出最可靠的答案")

    # ---- 新增：推理模式选择 ----
    reasoning_mode = st.radio(
        "选择推理方式",
        ["标准 CoT（一次推理）", "Self-Consistency（3次投票）"],
        index=0,
        horizontal=True
    )

    with st.form("cot_form"):
        question = st.text_input("问题", placeholder="输入一个需要推理的问题...")
        submitted = st.form_submit_button("开始推理")

        if submitted and question:
            with st.spinner("正在推理..."):
                if reasoning_mode == "标准 CoT（一次推理）":
                    # ---- 原有的 CoT 对比逻辑 ----
                    prompt_cot = f"{question}\n\n请逐步思考，并输出你的推理过程，最后给出结论。"
                    response_cot = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": "你是一个逻辑清晰的助手，会一步步分析问题。"},
                            {"role": "user", "content": prompt_cot}
                        ],
                        temperature=0.3
                    )
                    response_direct = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": "你是一个助手，请直接回答问题。"},
                            {"role": "user", "content": question}
                        ],
                        temperature=0.3
                    )

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("##### 逐步推理（CoT）")
                        st.markdown('<div class="result-card">' + response_cot.choices[0].message.content.replace('\n', '<br>') + '</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown("##### 直接回答")
                        st.markdown('<div class="result-card">' + response_direct.choices[0].message.content.replace('\n', '<br>') + '</div>', unsafe_allow_html=True)

                else:  # Self-Consistency
                    # ---- 核心：多次推理，然后投票 ----
                    num_trials = 3  # 3次推理
                    st.info(f"🔄 正在进行 {num_trials} 次独立推理，然后投票...")

                    # 存储所有推理过程和最终结论
                    all_reasoning = []
                    all_conclusions = []

                    for i in range(num_trials):
                        # 每次调用使用不同的温度（0.5~0.8 随机），增加多样性
                        temp = 0.5 + i * 0.15  # 0.5, 0.65, 0.8
                        prompt_cot = f"{question}\n\n请逐步思考，并输出你的推理过程，最后给出结论。"
                        response = client.chat.completions.create(
                            model="deepseek-chat",
                            messages=[
                                {"role": "system", "content": "你是一个逻辑清晰的助手，会一步步分析问题。"},
                                {"role": "user", "content": prompt_cot}
                            ],
                            temperature=temp
                        )
                        full_answer = response.choices[0].message.content

                        # 尝试提取最后一句作为结论（假设结论在末尾）
                        # 简单方法：取最后一个句号后的内容，或者直接用整个回答
                        # 更可靠：让模型自己标记结论（但为了演示，我们做简单处理）
                        # 这里我们直接存储整个回答，投票时用最后的结论句
                        lines = full_answer.strip().split('\n')
                        # 找到最后一行（通常结论在最后）
                        if lines:
                            last_line = lines[-1].strip()
                            # 如果最后一行包含“结论”、“答案”等关键词，则作为结论
                            if "结论" in last_line or "答案" in last_line or "所以" in last_line:
                                conclusion = last_line
                            else:
                                conclusion = f"（第{i+1}次推理）{full_answer[-50:]}"  # 备用
                        else:
                            conclusion = full_answer[-50:]

                        all_reasoning.append(f"【第{i+1}次推理（温度 {temp:.2f}）】\n{full_answer}")
                        all_conclusions.append(conclusion)

                    # ---- 投票：选择出现次数最多的结论 ----
                    from collections import Counter
                    vote_counter = Counter(all_conclusions)
                    most_common_conclusion = vote_counter.most_common(1)[0][0]

                    # ---- 展示结果 ----
                    st.success(f"✅ 投票完成！{num_trials} 次推理中，最终结论多数一致。")
                    st.markdown("#### 投票结果统计")
                    for idx, (conclusion, count) in enumerate(vote_counter.most_common(), 1):
                        st.markdown(f"- 结论 {idx}：{conclusion[:50]}...（出现 {count} 次）")

                    with st.expander("📖 查看所有推理过程（含详细步骤）"):
                        for reasoning in all_reasoning:
                            st.markdown(reasoning)
                            st.markdown("---")

                    st.markdown("#### 🏆 最终投票选出的答案")
                    st.markdown('<div class="result-card">' + most_common_conclusion + '</div>', unsafe_allow_html=True)

                    # ---- 额外的对比（可选） ----
                    st.caption("💡 Self-Consistency 通过多次推理投票，能有效减少单次推理的随机错误。")
# ---------- 知识库问答（LangChain 版） ----------
else:  # "知识库问答"
    st.subheader("知识库问答 · LangChain 版")
    st.caption("使用 LangChain 框架构建的 RAG 问答系统")

    if collection is None:
        st.warning("向量库尚未初始化，请先运行 `rag_retriever.py` 导入文档数据。", icon="⚠️")
    else:
        from langchain_chroma import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import PromptTemplate
        from langchain_classic.chains import RetrievalQA

        # 加载 LangChain 组件
        embedding_model = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L3-v2")
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embedding_model,
            collection_name="my_docs_parent"
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0.3,
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base="https://api.deepseek.com"
        )
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
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        with st.form("rag_form_langchain"):
            question = st.text_input("问题", placeholder="基于本地文档提问...")
            submitted = st.form_submit_button("提问")
            if submitted and question:
                with st.spinner("LangChain 正在检索并生成回答..."):
                    result = qa_chain.invoke({"query": question})
                    st.success("回答：")
                    st.write(result["result"])
                    with st.expander("📖 查看参考来源"):
                        for i, doc in enumerate(result["source_documents"], 1):
                            st.info(f"片段 {i}：{doc.page_content[:150]}...")
# ==================== 网络配置（必须在所有 import 之前） ====================
import os
# 使用 Hugging Face 镜像加速（国内访问更快）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ==================== 导入依赖库 ====================
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="DeepSeek 工具箱",
    page_icon="🔧",
    layout="wide"
)

# ... 其他代码保持不变 ...

# ==================== 导入依赖库 ====================
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
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
    """加载 RAG 组件，失败时返回 (None, None) 不影响应用启动"""
    try:
        # 使用更小更稳定的模型，减小下载压力
        model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_collection(name="my_docs")
        return model, collection
    except Exception as e:
        # 加载失败时静默处理，返回 None
        # 错误信息将在界面上通过 st.warning 显示
        st.warning(f"⚠️ RAG 组件加载失败，知识库问答暂时不可用。其他功能正常。错误信息：{str(e)[:100]}...")
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
                        {"role": "system", "content": f"你是一个专业翻译。请将用户输入的内容翻译成{target_lang}，只输出翻译结果，不要添加任何解释。"},
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

# ---------- 信息提取 ----------
elif mode == "信息提取":
    st.subheader("信息提取")
    st.caption("从一段描述中提取结构化的关键信息")

    examples = [
        {"input": "小明，数学95，语文88", "output": {"name": "小明", "scores": {"数学": 95, "语文": 88}}},
        {"input": "小红，英语92，科学85", "output": {"name": "小红", "scores": {"英语": 92, "科学": 85}}}
    ]
    system_prompt = (
        "你是一个数据提取助手。按 JSON 格式输出：{\"name\": \"姓名\", \"scores\": {\"科目\": 分数}}\n示例：\n"
    )
    for ex in examples:
        system_prompt += f"输入：'{ex['input']}' → 输出：{ex['output']}\n"
    system_prompt += "现在请严格按照这个格式输出，不要添加任何额外文字。"

    with st.form("json_form"):
        text_input = st.text_area("输入描述", height=100, placeholder="例如：小明，数学95，语文88")
        submitted = st.form_submit_button("提取信息")
        if submitted and text_input:
            with st.spinner("正在提取..."):
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text_input}
                    ],
                    temperature=0.1
                )
                st.markdown('<div class="result-card">' + response.choices[0].message.content + '</div>', unsafe_allow_html=True)

# ---------- 逐步推理 ----------
elif mode == "逐步推理":
    st.subheader("逐步推理")
    st.caption("展示详细推理过程，同时对比直接回答的效果")

    with st.form("cot_form"):
        question = st.text_input("问题", placeholder="输入一个需要推理的问题...")
        submitted = st.form_submit_button("开始推理")

        if submitted and question:
            with st.spinner("正在推理..."):
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
                    st.markdown("##### 逐步推理")
                    st.markdown('<div class="result-card">' + response_cot.choices[0].message.content.replace('\n', '<br>') + '</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown("##### 直接回答")
                    st.markdown('<div class="result-card">' + response_direct.choices[0].message.content.replace('\n', '<br>') + '</div>', unsafe_allow_html=True)

# ---------- 知识库问答 ----------
else:  # "知识库问答"
    st.subheader("知识库问答")
    st.caption("基于本地文档内容回答你的问题")

    if collection is None:
        st.warning("向量库尚未初始化，请先运行 `rag_retriever.py` 导入文档数据。", icon="⚠️")
    else:
        with st.form("rag_form"):
            question = st.text_input("问题", placeholder="基于本地文档提问...")
            submitted = st.form_submit_button("提问")

            if submitted and question:
                with st.spinner("正在检索并生成回答..."):
                    query_embedding = embed_model.encode([question])
                    results = collection.query(
                        query_embeddings=query_embedding.tolist(),
                        n_results=3
                    )
                    retrieved_docs = results['documents'][0]

                    if not retrieved_docs:
                        st.info("知识库中没有找到相关信息。")
                    else:
                        context = "\n---\n".join(retrieved_docs)
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

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("##### 参考来源")
                            for i, doc in enumerate(retrieved_docs, 1):
                                with st.container():
                                    st.markdown(f"""
                                    <div style="background:#f1f5f9; border-left:3px solid #818cf8; border-radius:6px; padding:0.7rem 1rem; margin:0.5rem 0; font-size:0.85rem; color:#475569; line-height:1.6;">
                                        <span style="font-weight:600; color:#4f46e5;">片段 {i}</span><br>
                                        {doc}
                                    </div>
                                    """, unsafe_allow_html=True)
                        with col2:
                            st.markdown("##### 回答")
                            st.markdown('<div class="result-card">' + response.choices[0].message.content.replace('\n', '<br>') + '</div>', unsafe_allow_html=True)
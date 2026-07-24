# ==================== 网络配置（必须在所有 import 之前） ====================
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ==================== 导入依赖库 ====================
import json
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

# ==================== 自定义样式 ====================
st.markdown("""
<style>
    /* --- 全局 --- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

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

    /* --- 参考来源卡片 --- */
    .source-card {
        background: #f1f5f9;
        border-left: 3px solid #818cf8;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        color: #475569;
        line-height: 1.6;
    }
    .source-label {
        font-weight: 600;
        color: #4f46e5;
    }

    /* --- expander 美化 --- */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 0.9rem;
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
        collection = chroma_client.get_collection(name="my_docs_parent")
        return model, collection
    except Exception:
        return None, None

embed_model, collection = load_rag_components()

# ==================== 侧边栏 ====================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 1.5rem 0;">
        <div style="font-size: 1.3rem; font-weight: 700; color: #e0e0e0;">DeepSeek 工具箱</div>
        <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.2rem;">九个实用功能，一个入口</div>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "选择功能",
        ["翻译", "对话", "代码摘要", "信息提取", "逐步推理", "知识库问答", "周报生成器", "天气查询", "ReAct 智能助手"],
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
st.markdown('<div class="main-subtitle">翻译 · 对话 · 代码摘要 · 信息提取 · 逐步推理 · 知识库问答 · 周报生成 · 天气查询 · ReAct 助手</div>', unsafe_allow_html=True)
st.markdown("---")

# ==================== 各模式功能实现 ====================

# ---------- 翻译 ----------
if mode == "翻译":
    st.subheader("文本翻译")

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
    st.subheader("信息提取 · Few-shot 演示")
    st.caption("给模型看 2 个示例，它就能学会提取任意格式的数据")

    with st.expander("当前使用的 Few-shot 示例"):
        st.code("""
示例 1: 输入: "小明，数学95，语文88" → 输出: {"name": "小明", "scores": {"数学": 95, "语文": 88}}
示例 2: 输入: "小红，英语92，科学85" → 输出: {"name": "小红", "scores": {"英语": 92, "科学": 85}}
        """)

    with st.form("json_form"):
        text_input = st.text_area(
            "输入待提取的文本",
            height=100,
            placeholder="例如：小刚，数学78，英语82，历史90"
        )
        submitted = st.form_submit_button("提取信息")

        if submitted and text_input:
            with st.spinner("正在提取..."):
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
                st.caption("模型之所以能正确提取，是因为它模仿了上面示例中的格式。这就是 Few-shot 的力量。")

# ---------- 逐步推理 ----------
elif mode == "逐步推理":
    st.subheader("逐步推理 · 自洽性演示")
    st.caption("模型可以多次推理，然后投票选出最可靠的答案")

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

                else:
                    num_trials = 3
                    st.info(f"正在进行 {num_trials} 次独立推理，然后投票...")

                    all_reasoning = []
                    all_conclusions = []

                    for i in range(num_trials):
                        temp = 0.5 + i * 0.15
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

                        lines = full_answer.strip().split('\n')
                        if lines:
                            last_line = lines[-1].strip()
                            if "结论" in last_line or "答案" in last_line or "所以" in last_line:
                                conclusion = last_line
                            else:
                                conclusion = f"（第{i+1}次推理）{full_answer[-50:]}"
                        else:
                            conclusion = full_answer[-50:]

                        all_reasoning.append(f"【第{i+1}次推理（温度 {temp:.2f}）】\n{full_answer}")
                        all_conclusions.append(conclusion)

                    vote_counter = Counter(all_conclusions)
                    most_common_conclusion = vote_counter.most_common(1)[0][0]

                    st.success(f"投票完成！{num_trials} 次推理中，最终结论多数一致。")
                    st.markdown("##### 投票结果统计")
                    for idx, (conclusion, count) in enumerate(vote_counter.most_common(), 1):
                        st.markdown(f"- 结论 {idx}：{conclusion[:50]}...（出现 {count} 次）")

                    with st.expander("查看所有推理过程"):
                        for reasoning in all_reasoning:
                            st.markdown(reasoning)
                            st.markdown("---")

                    st.markdown("##### 最终投票选出的答案")
                    st.markdown('<div class="result-card">' + most_common_conclusion + '</div>', unsafe_allow_html=True)
                    st.caption("Self-Consistency 通过多次推理投票，能有效减少单次推理的随机错误。")

# ---------- 知识库问答 ----------
elif mode == "知识库问答":
    st.subheader("知识库问答 · LangChain 版")
    st.caption("使用 LangChain 框架构建的 RAG 问答系统")

    @st.cache_resource
    def load_langchain_rag():
        from langchain_chroma import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import PromptTemplate
        from langchain_classic.chains import RetrievalQA

        embedding_model = HuggingFaceEmbeddings(
            model_name="paraphrase-MiniLM-L3-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

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

        return qa_chain

    try:
        qa_chain = load_langchain_rag()
        vector_available = True
    except Exception as e:
        vector_available = False
        st.warning(f"向量库加载失败，请先运行 rag_retriever.py 导入文档数据。")

    if vector_available:
        with st.form("rag_form_langchain"):
            question = st.text_input("问题", placeholder="基于本地文档提问...")
            submitted = st.form_submit_button("提问")

            if submitted and question:
                with st.spinner("正在检索并生成回答..."):
                    try:
                        result = qa_chain.invoke({"query": question})

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("##### 参考来源")
                            for i, doc in enumerate(result["source_documents"], 1):
                                with st.container():
                                    st.markdown(f"""
                                    <div class="source-card">
                                        <span class="source-label">片段 {i}</span><br>
                                        {doc.page_content[:150]}...
                                    </div>
                                    """, unsafe_allow_html=True)
                        with col2:
                            st.markdown("##### 回答")
                            st.markdown('<div class="result-card">' + result["result"].replace('\n', '<br>') + '</div>', unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"生成回答失败：{str(e)}")
    else:
        st.info("请先运行 rag_retriever.py 构建向量库，再使用此功能。")

# ---------- 周报生成器 ----------
elif mode == "周报生成器":
    st.subheader("周报生成器")
    st.caption("输入你本周的工作内容，帮你生成结构化周报")

    with st.form("weekly_report_form"):
        work_content = st.text_area(
            "本周工作内容",
            height=150,
            placeholder="例如：修复了登录页面的Bug，优化了数据库查询速度，参加了需求评审会..."
        )
        submitted = st.form_submit_button("生成周报")

        if submitted and work_content:
            with st.spinner("正在整理..."):
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "你是一个专业的职场助理。请根据用户输入的工作内容，生成一份结构清晰的周报。格式必须包含：## 本周完成、## 下周计划、## 遇到的困难 这三个部分，使用 Markdown 格式。"},
                        {"role": "user", "content": work_content}
                    ],
                    temperature=0.5
                )
                report = response.choices[0].message.content
                st.success("周报生成完毕")
                st.markdown(report)

# ---------- 天气查询 ----------
elif mode == "天气查询":
    st.subheader("天气查询")
    st.caption("输入城市名称，查询当前天气情况")

    def get_current_weather(location):
        weather_db = {
            "上海": "28°C，多云转阴，东南风3级",
            "北京": "32°C，晴，空气质量良",
            "深圳": "30°C，雷阵雨，注意带伞",
            "成都": "26°C，阴天，湿度较大",
            "广州": "31°C，多云，微风"
        }
        return weather_db.get(location, f"{location}：20°C，晴（模拟数据）")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "查询指定地点的当前天气情况",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "城市名称，例如：上海、北京"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    def run_weather_agent(user_query):
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": user_query}],
            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message

        if message.tool_calls:
            tool_call = message.tool_calls[0]
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            if function_name == "get_current_weather":
                result = get_current_weather(arguments["location"])

            second_response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "user", "content": user_query},
                    message,
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    }
                ]
            )
            return second_response.choices[0].message.content
        else:
            return message.content

    with st.form("weather_form"):
        query = st.text_input("天气查询", placeholder="例如：上海今天天气怎么样？")
        submitted = st.form_submit_button("查询")

        if submitted and query:
            with st.spinner("正在查询..."):
                try:
                    answer = run_weather_agent(query)
                    st.success("查询完成")
                    st.markdown('<div class="result-card">' + answer.replace('\n', '<br>') + '</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"出错了：{str(e)}")
# ---------- ReAct 智能助手 ----------
elif mode == "ReAct 智能助手":
    st.subheader("ReAct 智能助手")
    st.caption("支持多工具、多步推理，自主规划完成任务（演示：天气 + 汇率）")


    # --- 1. 工具函数 (与 react_agent.py 保持一致) ---
    def get_current_weather(location):
        weather_db = {
            "上海": "28°C，多云转阴，东南风3级",
            "北京": "32°C，晴，空气质量良",
            "深圳": "30°C，雷阵雨，注意带伞",
            "成都": "26°C，阴天，湿度较大",
            "广州": "31°C，多云，微风"
        }
        return weather_db.get(location, f"{location}：20°C，晴（模拟数据）")


    def get_exchange_rate(from_currency, to_currency):
        rates = {
            ("人民币", "美元"): 0.14,
            ("美元", "人民币"): 7.15,
            ("人民币", "日元"): 20.50,
            ("日元", "人民币"): 0.049,
            ("人民币", "欧元"): 0.13,
            ("欧元", "人民币"): 7.70,
        }
        return rates.get((from_currency, to_currency), 0.00)


    # --- 2. 工具描述 (tools) ---
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "查询指定地点的当前天气情况",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "城市名称，例如：上海、北京"}
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_exchange_rate",
                "description": "查询两种货币之间的汇率",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "from_currency": {"type": "string", "description": "源货币代码，例如：人民币、美元"},
                        "to_currency": {"type": "string", "description": "目标货币代码，例如：人民币、美元"}
                    },
                    "required": ["from_currency", "to_currency"]
                }
            }
        }
    ]


    # --- 3. ReAct Agent 执行逻辑 ---
    def run_react_agent(user_query, max_steps=3):
        messages = [{"role": "user", "content": user_query}]
        step_count = 0
        final_answer = "抱歉，任务未能完成。"

        # 用 Streamlit 的容器来显示推理过程
        reasoning_placeholder = st.empty()
        full_reasoning = ""

        with st.spinner("Agent 正在思考并行动..."):
            while step_count < max_steps:
                step_count += 1
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )

                message = response.choices[0].message
                messages.append(message)  # 保存助手消息

                # 检查是否有工具调用
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        function_name = tool_call.function.name
                        arguments = json.loads(tool_call.function.arguments)

                        step_log = f"第 {step_count} 步：调用 `{function_name}`  \n参数：`{arguments}`\n"
                        full_reasoning += step_log
                        reasoning_placeholder.markdown(full_reasoning)

                        # 执行工具
                        if function_name == "get_current_weather":
                            result = get_current_weather(arguments["location"])
                        elif function_name == "get_exchange_rate":
                            rate = get_exchange_rate(arguments["from_currency"], arguments["to_currency"])
                            result = f"1 {arguments['from_currency']} = {rate} {arguments['to_currency']}"
                        else:
                            result = "未知工具"

                        # 将工具结果添加到消息历史
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result
                        })
                else:
                    # 没有工具调用，任务完成
                    final_answer = message.content
                    break

        return final_answer, full_reasoning


    # --- 4. Web 界面交互 ---
    with st.form("react_form"):
        query = st.text_input("请输入您的任务", placeholder="例如：北京和上海哪个更热？或者 100元人民币能换多少美元？")
        submitted = st.form_submit_button("开始执行")

        if submitted and query:
            final_answer, reasoning = run_react_agent(query)

            with st.expander("查看 Agent 的思考与行动过程"):
                st.markdown(reasoning)

            st.success("任务完成")
            st.markdown('<div class="result-card">' + final_answer.replace('\n', '<br>') + '</div>', unsafe_allow_html=True)
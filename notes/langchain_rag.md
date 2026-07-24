# LangChain RAG 实践笔记

**日期：** 2026-07-21

## 一、背景与目标

在完成手写 RAG（HyDE + Re-ranking + 父子文档检索）后，我决定用 **LangChain 框架** 重构 RAG 链路。目标是：
- 减少重复代码，提高可维护性
- 学习主流大模型应用框架的使用
- 在简历中增加“LangChain”技术栈

## 二、实践内容

### 1. 安装依赖

```bash
pip install langchain langchain-community langchain-chroma langchain-openai
```

### 2. 核心代码结构

```python
# 1. 创建 Embedding 模型
embedding_model = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L3-v2")

# 2. 加载 Chroma 向量库
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model,
    collection_name="my_docs_parent"
)

# 3. 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. 创建 LLM（DeepSeek）
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.3,
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com"
)

# 5. 构建 Prompt 模板
template = """你是一个知识库问答助手。请根据【参考内容】回答用户的问题...
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# 6. 构建 RAG 链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# 7. 调用
result = qa_chain.invoke({"query": "什么是 RAG？"})
```

## 三、手写 RAG vs LangChain RAG 对比

| 对比项 | 手写 RAG | LangChain RAG |
| :--- | :--- | :--- |
| **代码量** | 100+ 行 | ~50 行 |
| **可维护性** | 较低 | 高（模块化） |
| **可扩展性** | 需大量改动 | 只需替换组件 |
| **学习成本** | 低 | 中等 |
| **面试价值** | 能讲原理 | 能讲原理 + 框架经验 |

## 四、核心收获

1. **LangChain 的核心抽象**
   - `Retriever`：负责检索
   - `LLM`：负责生成
   - `Chain`：将多个步骤串联
   - `PromptTemplate`：管理提示词

2. **从“手写”到“框架”的思考**
   - 框架解决的是“重复造轮子”的问题
   - 理解底层原理后使用框架，才能灵活调优

3. **兼容性注意事项**
   - LangChain 1.x 版本中，`PromptTemplate` 从 `langchain_core.prompts` 导入
   - `RetrievalQA` 从 `langchain_classic.chains` 或 `langchain.chains` 导入
   - 不同版本 API 有差异，需根据实际环境调整

## 五、集成到 Web 界面

将 LangChain RAG 集成到 `app.py` 的“知识库问答”模式中，使用 `@st.cache_resource` 缓存 `qa_chain`，避免每次请求都重新加载模型。

## 六、面试话术

> “在完成手写 RAG 之后，我使用 LangChain 框架进行了重构。LangChain 提供了一套标准化的接口，让我能够用更少的代码实现更清晰的逻辑。目前这个 RAG 系统同时支持 HyDE、Re-ranking 和父子文档检索，已集成到 Web 应用中并部署上线。”

## 七、下一阶段目标

- 学习 LangChain 的 Agent 和 Tool 机制
- 实现 Function Calling 和 ReAct 模式
- 用 LangGraph 构建多步 Agent 工作流

## 八、参考资料

- [LangChain 官方文档](https://python.langchain.com/)
- [LangChain Chroma 集成](https://python.langchain.com/docs/integrations/vectorstores/chroma/)
- [DeepSeek API 文档](https://platform.deepseek.com/api-docs/)
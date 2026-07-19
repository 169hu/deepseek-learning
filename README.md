# DeepSeek 多功能工具箱

基于 DeepSeek API 构建的 6 合 1 智能工具，支持翻译、对话、代码摘要、信息提取、思维链推理和 RAG 知识库问答。

## ✨ 功能列表

| 功能 | 说明 |
| :--- | :--- |
| 🌍 **翻译** | 支持中英日韩法德互译 |
| 💬 **自由对话** | 带上下文的连续聊天 |
| 📄 **代码摘要** | 快速理解代码功能 |
| 📊 **信息提取** | 从文本中提取结构化 JSON 数据 |
| 🧠 **逐步推理** | 展示思维链，对比直接回答 |
| 📚 **知识库问答** | 基于本地文档的 RAG 问答 |

## 🛠️ 技术栈

- **前端**：Streamlit
- **LLM**：DeepSeek API
- **向量库**：ChromaDB + sentence-transformers
- **语言**：Python 3.10+

## 🚀 快速开始

```bash
# 1. 克隆项目
git clone https://github.com/169hu/deepseek-learning.git
cd deepseek-learning

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置密钥
cp .env.example .env
# 编辑 .env，填入你的 DEEPSEEK_API_KEY

# 4. 构建知识库（可选，RAG 模式需要）
python rag_retriever.py

# 5. 启动 Web 界面
streamlit run app.py
#6. 📸 效果截图

## 主界面
![主界面](images/main.png)

## RAG 知识库问答演示
![RAG 问答演示](images/demo.png)
## 👨‍💻 作者

**胡进林** · 计算机科学与技术专业 · 大三

- 📧 邮箱：161725862@qq.com
- 🔗 GitHub：[169hu](https://github.com/169hu)
- 📍 目标岗位：大模型应用开发实习生
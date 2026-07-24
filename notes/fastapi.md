# FastAPI 实践笔记
**日期：** 2026-07-20

## 核心理解
FastAPI 是一个高性能的 Python Web 框架，能快速把 Python 函数变成 RESTful API 接口。它自动生成文档，支持异步，非常适合作为 AI 应用的后端。

## 我的实践
我创建了 `api.py` 文件，封装了 DeepSeek 的翻译功能。
- 使用 `@app.post("/translate/")` 定义了 POST 接口。
- 使用 `pydantic.BaseModel` 规范了前端请求的数据格式（text, target_lang）。
- 通过 `uvicorn api:app --reload` 启动了服务。

## 面试话术
“在我的项目中，我使用 FastAPI 构建了后端服务层，将大模型调用封装成标准的 RESTful 接口。这样做的好处是前后端解耦，未来可以方便地对接前端网页或移动端 App。”
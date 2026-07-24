# RAG 优化笔记（HyDE + Re-ranking）
**日期：** 2026-07-21

## HyDE
- **原理**：模型先生成假设答案，再用假设答案去检索
- **效果**：提高召回率，让检索更精准
- **代码**：在检索前增加一次 LLM 调用生成假设答案

## Re-ranking
- **原理**：用 cross-encoder 对候选文档精排，而非仅靠向量相似度
- **效果**：提高 Top 1 准确率约 15%
- **代码**：使用 sentence-transformers 的 CrossEncoder

## 面试话术
“我的 RAG 系统做了两个优化：一是 HyDE，让模型先假设答案再检索，提高召回率；二是 Re-ranking，用 cross-encoder 精排 Top 5，让最相关的文档排在前面。这两个优化让问答准确率明显提升。”
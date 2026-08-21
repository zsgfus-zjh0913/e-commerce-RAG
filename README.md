# 电商 RAG 知识库系统

基于 Flask + TF-IDF + DeepSeek API 的电商智能客服系统，支持商品信息检索、尺码查询、文档上传更新知识库。

## 功能

- 智能问答：支持自然语言提问，自动检索知识库并生成回答
- 尺码查询：支持口语化表达（如"男士一米八穿多大码"）
- 文档管理：侧边栏上传 `.txt/.md/.csv/.json/.docx` 文件，自动重建索引
- 双模式：有 API Key 时走 LLM 流式回答，无 Key 时返回纯检索结果
- LLM 回退：API 失败时自动回退到检索结果

## 技术栈

| 层面 | 技术 |
|------|------|
| 后端 | Flask 3.x |
| 检索 | scikit-learn TF-IDF + 余弦相似度 |
| 中文分词 | jieba（含电商领域词典） |
| 大模型 | DeepSeek API（流式输出） |
| 文档解析 | python-docx |
| 前端 | 原生 HTML/CSS/JS |

## 快速开始

```bash
pip install -r requirements.txt
python app.py
```

访问 http://127.0.0.1:5000

## 项目结构

```
├── app.py              # Flask 后端
├── rag_engine.py       # RAG 检索引擎
├── llm_client.py       # DeepSeek API 客户端
├── requirements.txt
├── data/               # 知识库文档
│   ├── products.json   # 商品信息
│   ├── size_chart.csv  # 尺码对照表
│   ├── faq.txt         # 常见问题
│   ├── shipping_policy.md
│   └── 1.docx          # 尺码表文档
├── templates/
│   └── index.html      # 网页界面
└── static/
    ├── css/style.css
    └── js/main.js
```

## 配置 API Key

在网页侧边栏「AI 模型设置」中输入 DeepSeek API Key，获取地址：https://platform.deepseek.com/api_keys

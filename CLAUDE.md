# SpaceX 股票分析问答助手 — 项目规范

## 项目概述

构建一个基于 RAG（检索增强生成）的 SpaceX 股票分析问答助手。
- **前端**：Streamlit 聊天界面
- **后端**：LangChain RAG Pipeline
- **LLM**：Claude API（`claude-sonnet-4-6`）
- **向量库**：Chroma（本地持久化）
- **Embedding**：`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`（支持中英文）

---

## 目录结构

```
spacex-analyst/
├── CLAUDE.md                  ← 本文件，项目总规范
├── prompts/
│   ├── system.txt             ← LLM 系统角色设定
│   ├── qa_chain.txt           ← RAG 问答合成提示词
│   ├── retrieval.txt          ← 检索 Query 改写提示词
│   └── classifier.txt         ← 问题分类提示词（可选路由）
├── docs/                      ← 放置知识库原始文档（PDF / TXT / MD）
├── chroma_db/                 ← Chroma 向量库持久化目录（自动生成）
├── app.py                     ← Streamlit 前端入口
├── rag_pipeline.py            ← RAG 核心逻辑
├── ingest.py                  ← 文档摄取脚本（读取 docs/ 并写入向量库）
├── config.py                  ← 统一配置（模型名、路径、参数）
├── requirements.txt
└── .env.example               ← 环境变量模板
```

---

## 核心规范（必须遵守）

### 提示词规范
- **所有提示词只存放在 `prompts/` 目录**，`.py` 文件中通过读取文件加载，不允许硬编码提示词字符串进业务逻辑
- 修改提示词只改 `prompts/` 目录下的 `.txt` 文件
- 提示词文件使用 UTF-8 编码

### 代码规范
- Python 3.10+，类型注解（`typing`）
- 配置项统一在 `config.py` 管理，不散落在各文件
- 环境变量通过 `python-dotenv` 加载，API Key 不得出现在代码中
- 所有函数必须有 docstring

### LangChain 规范
- 使用 `ConversationalRetrievalChain`，支持多轮对话
- 检索器使用 `MMR`（最大边际相关性），`k=5`，减少重复文档
- `return_source_documents=True`，在前端展示来源
- Embedding 模型本地加载，不调用 OpenAI

### Streamlit 规范
- 使用 `st.session_state` 管理对话历史
- 每条回答下方展示折叠的「参考来源」（文档名 + 相关片段）
- 侧边栏提供：知识库状态、清除对话、重新加载文档 功能
- 加载时显示 spinner，不允许页面卡死无反馈

---

## 分步构建顺序

Claude Code 请按以下顺序执行，每步完成后等待确认：

1. **Step 1**：创建 `config.py`、`.env.example`、`requirements.txt`
2. **Step 2**：将 `prompts/` 目录下四个提示词文件写入（内容见下方）
3. **Step 3**：实现 `ingest.py`（文档加载 → 切片 → 向量化 → 写入 Chroma）
4. **Step 4**：实现 `rag_pipeline.py`（加载向量库 → 构建 Chain）
5. **Step 5**：实现 `app.py`（Streamlit UI）
6. **Step 6**：整体联调，确保 `python ingest.py` 和 `streamlit run app.py` 均可运行

---

## 提示词内容（写入 prompts/ 目录）

### prompts/system.txt
```
你是一位专业的股票分析助手，专注于 SpaceX（Space Exploration Technologies Corp.）的投资研究。

你的能力范围：
- 基于知识库中的文档、财报、新闻、行业报告回答用户问题
- 分析 SpaceX 的商业模式、竞争优势与核心收入来源（卫星发射服务、Starlink 宽带、政府合同、星舰研发）
- 解读火箭发射频率、复用率、Starlink 用户增长对公司估值的影响
- 对比竞争对手分析（ULA、Blue Origin、Rocket Lab、中国长征系列）
- 解释相关财务指标（ARR、EBITDA、估值倍数、融资轮次）和风险因素

你的行为规范：
- 仅基于检索到的文档内容作答，不凭空捏造数据或事件
- 若知识库中没有足够信息，明确告知：「当前知识库中未找到相关内容，建议查阅最新资料」
- 所有数据和事件引用须注明来源文档名称和时间（如有）
- 分析结论须区分「事实」与「判断/预测」，预测性内容使用「预计」「据分析」等措辞
- 涉及投资观点时，必须在回答末尾附上免责声明
- 回答语言：中文为主，专业术语保留英文原文（如 EBITDA、Starlink、Falcon 9）

免责声明（每次涉及投资观点时附加）：
⚠️ 以上内容仅基于现有文档资料整理，不构成任何投资建议。投资决策请结合专业财务顾问意见，自行承担风险。
```

### prompts/qa_chain.txt
```
你是 SpaceX 投资研究专家。请严格根据以下检索到的背景文档内容，回答用户的问题。

背景文档：
{context}

对话历史：
{chat_history}

用户当前问题：{question}

回答要求：
1. 结构化输出：先给出核心结论（1-2句），再展开分析依据
2. 优先引用文档中的具体数据、时间节点和事件，格式为：「根据[来源]显示，...」
3. 若多份文档观点不一致，需对比说明差异并给出综合判断
4. 不确定或文档未覆盖的内容，使用「据报道」「预计」等措辞，不得伪造数据
5. 涉及竞争对比时，保持客观中立，不带主观倾向
6. 回答长度：普通问题 200-400 字，复杂分析题可延伸至 600 字
7. 若涉及投资判断，在末尾附加免责声明

请用中文回答：
```

### prompts/retrieval.txt
```
你的任务是将用户的问题改写为适合向量数据库检索的查询语句，以提高文档召回率。

用户原始问题：{user_question}

请生成 3 个不同角度的检索查询：
1. 提取核心关键词，去除口语化表达
2. 翻译为英文版本（用于英文文档检索）
3. 扩展同义词和相关概念版本

输出要求：仅输出 JSON 格式，不要有其他文字：
{
  "queries": ["中文关键词版本", "English keyword version", "扩展同义词版本"]
}
```

### prompts/classifier.txt
```
判断以下用户问题最匹配的分析类别，仅输出 JSON，不要有其他文字。

分类选项：
1 - 财务数据（收入、估值、融资、利润）
2 - 业务进展（发射次数、Starlink 数据、新合同）
3 - 竞争分析（对比 Blue Origin、Rocket Lab、ULA 等）
4 - 风险评估（监管、技术故障、市场风险）
5 - 宏观环境（政策、利率、航天市场趋势）
6 - 其他

用户问题：{question}

输出格式：
{"category": 数字, "reason": "一句话说明分类依据"}
```

---

## config.py 核心配置参考

```python
# 模型
LLM_MODEL = "claude-sonnet-4-6"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# 路径
CHROMA_DB_PATH = "./chroma_db"
DOCS_PATH = "./docs"
PROMPTS_PATH = "./prompts"

# 检索参数
RETRIEVER_K = 5
RETRIEVER_FETCH_K = 20  # MMR 候选池大小
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# 对话历史保留轮数
MAX_HISTORY_TURNS = 6
```

---

## requirements.txt 核心依赖参考

```
anthropic>=0.25.0
langchain>=0.2.0
langchain-anthropic>=0.1.0
langchain-community>=0.2.0
langchain-chroma>=0.1.0
sentence-transformers>=2.7.0
streamlit>=1.35.0
python-dotenv>=1.0.0
chromadb>=0.5.0
pypdf>=4.0.0
unstructured>=0.13.0
```

---

## 知识库文档建议（放入 docs/ 目录）

| 文档类型 | 建议来源 |
|----------|----------|
| SpaceX 历年发射记录 | Wikipedia 导出 / SpaceX 官网新闻 |
| Starlink 用户与收入报告 | 分析师报告、Bloomberg 新闻 |
| SpaceX 估值与融资历史 | Crunchbase、PitchBook 导出 |
| 竞争对手对比报告 | 行业研报 PDF |
| 政府合同公告 | NASA.gov、DoD 新闻稿 |
| 马斯克/管理层公开讲话 | 财经媒体整理文章 |

---

## 联调验证检查点

- [ ] `python ingest.py` 无报错，Chroma 目录生成
- [ ] `streamlit run app.py` 页面正常打开
- [ ] 输入问题后能返回带来源的答案
- [ ] 对话历史在多轮对话中正确传递
- [ ] 侧边栏「清除对话」功能正常
- [ ] 知识库为空时有友好提示

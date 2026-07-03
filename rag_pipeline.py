"""
RAG 核心逻辑 —— 提供两种管道：
1. build_rag_chain()    → 传统 ConversationalRetrievalChain（仅知识库）
2. build_agent()        → 🆕 Tool-Calling Agent（知识库 + 联网搜索 + 股价查询）
"""

from typing import Tuple, Any, Dict

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import StructuredTool

from config import (
    LLM_MODEL,
    GROQ_API_KEY,
    GROQ_BASE_URL,
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
    RETRIEVER_K,
    RETRIEVER_FETCH_K,
    MAX_HISTORY_TURNS,
    ENABLE_WEB_SEARCH,
    ENABLE_STOCK_LOOKUP,
    load_prompt,
    get_embeddings,
)


def _create_llm(temperature: float = 0.3) -> ChatOpenAI:
    """创建 Groq LLM 实例（兼容 OpenAI SDK，免费 tier）。

    Args:
        temperature: 生成温度，分析场景用较低值保证一致性

    Returns:
        ChatOpenAI 实例，指向 Groq API
    """
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
        temperature=temperature,
        max_tokens=2048,
    )


def _create_embeddings():
    """创建免费本地 FastEmbed 实例。

    Returns:
        FastEmbedEmbeddings 实例，无需 API Key
    """
    return get_embeddings()


def _load_vectorstore() -> Chroma:
    """加载本地 Chroma 向量库。

    若目录不存在则自动创建空向量库（Agent 仍可通过联网搜索工作）。

    Returns:
        Chroma 向量库实例
    """
    import os
    embeddings = _create_embeddings()

    if not os.path.isdir(CHROMA_DB_PATH):
        # 自动创建空向量库
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_PATH,
        )
        # 写入一条占位文档以完成初始化（避免后续加载出问题）
        from langchain_core.documents import Document
        vectorstore.add_documents([
            Document(page_content="[知识库待初始化]", metadata={"source": "_placeholder"})
        ])
        return vectorstore

    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )


def _build_qa_prompt() -> PromptTemplate:
    """构建 QA 链提示词模板（系统角色 + 问答指令）。

    从 prompts/system.txt 和 prompts/qa_chain.txt 合并生成。

    Returns:
        PromptTemplate 实例，含 {context}, {chat_history}, {question} 占位符
    """
    system_prompt = load_prompt("system.txt")
    qa_template = load_prompt("qa_chain.txt")

    # 合并：系统角色在前，问答指令在后
    full_template = f"{system_prompt}\n\n---\n\n{qa_template}"

    return PromptTemplate(
        template=full_template,
        input_variables=["context", "chat_history", "question"],
    )


# ============================================================
# 传统 Chain 管道（向后兼容）
# ============================================================

def build_rag_chain() -> Tuple[ConversationalRetrievalChain, Any]:
    """构建 RAG 对话检索链（仅知识库，无实时数据）。

    使用 MMR 检索器（最大边际相关性），减少重复文档，提高答案覆盖度。

    Returns:
        (chain, retriever) 元组
        - chain: ConversationalRetrievalChain 实例
        - retriever: MMR 检索器（供前端展示来源时使用）

    Raises:
        FileNotFoundError: 向量库不存在
        ValueError: API Key 未配置
    """
    # 校验 API Key
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY 未设置。\n"
            "免费获取：https://console.groq.com → 注册 → API Keys\n"
            "本地开发：写入 .env 文件 → GROQ_API_KEY=gsk_xxxxx\n"
            "Streamlit Cloud：Manage app → Secrets → 填入 TOML 格式密钥后 Reboot"
        )

    # 1. 加载向量库
    vectorstore = _load_vectorstore()

    # 2. 构建 MMR 检索器
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RETRIEVER_K,
            "fetch_k": RETRIEVER_FETCH_K,
        },
    )

    # 3. 创建 LLM
    llm = _create_llm()

    # 4. 构建 QA Prompt
    qa_prompt = _build_qa_prompt()

    # 5. 组装 ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        max_tokens_limit=4000,
    )

    return chain, retriever


# ============================================================
# 🆕 Agent 管道（知识库 + 联网搜索 + 股价查询）
# ============================================================

def _retrieve_knowledge(query: str) -> str:
    """搜索本地 Chroma 知识库，获取 SpaceX 相关文档片段。

    此函数被 Agent 作为 Tool 调用，每次调用都会重新创建检索器。
    （通过闭包捕获 _retriever_ref）
    """
    retriever = _retriever_ref.get("retriever")
    if retriever is None:
        return "[错误] 知识库检索器未初始化，请检查向量库状态。"

    try:
        docs = retriever.invoke(query)
    except Exception as e:
        return f"[检索失败] {e}"

    if not docs:
        return "[无结果] 知识库中未找到相关内容。"

    lines = [f"=== 知识库检索结果: 「{query}」 ==="]
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知")
        page = doc.metadata.get("page", -1)
        content = doc.page_content[:500]
        page_info = f"第 {page + 1} 页" if page >= 0 else ""
        lines.append(f"\n--- 文档 {i}: {source} {page_info} ---")
        lines.append(content)

    return "\n".join(lines)


# 模块级引用，用于 _retrieve_knowledge 访问当前检索器
_retriever_ref: Dict[str, Any] = {}


def _create_agent_tools() -> list:
    """创建 Agent 工具列表：知识库检索 + 联网搜索 + 股价查询。

    Returns:
        LangChain StructuredTool 列表
    """
    tools = []

    # Tool 1: 知识库检索（始终启用）
    kb_tool = StructuredTool.from_function(
        func=_retrieve_knowledge,
        name="retrieve_knowledge",
        description=(
            "搜索本地 SpaceX 知识库，获取文档中的历史数据、分析报告、商业模式、技术细节等。"
            "适用于：SpaceX 基本业务、历史发射数据、Starlink 技术、竞争对手分析框架等非时效性问题。"
            "输入：自然语言查询语句（中文或英文）"
        ),
    )
    tools.append(kb_tool)

    # Tool 2: 联网搜索
    if ENABLE_WEB_SEARCH:
        from tools import search_web
        web_tool = StructuredTool.from_function(
            func=search_web,
            name="search_web",
            description=(
                "搜索互联网获取最新信息。当用户询问时效性问题（如最新估值、最近发射、新闻动态）"
                "或知识库中没有足够信息时使用。"
                "输入：搜索关键词（中文或英文）"
            ),
        )
        tools.append(web_tool)

    # Tool 3: 股价查询
    if ENABLE_STOCK_LOOKUP:
        from tools import get_stock_price
        stock_tool = StructuredTool.from_function(
            func=get_stock_price,
            name="get_stock_price",
            description=(
                "查询航天/国防相关上市公司实时股价。支持美股代码如 RKLB (Rocket Lab)、"
                "LUNR (Intuitive Machines)、BA (Boeing)、LMT (Lockheed Martin)、"
                "NOC (Northrop Grumman)、SPCE (Virgin Galactic) 等。"
                "输入：单个股票代码如 'RKLB'，或多个逗号分隔如 'RKLB,LUNR,BA'"
            ),
        )
        tools.append(stock_tool)

    return tools


def _create_agent_prompt() -> ChatPromptTemplate:
    """构建 Agent 的 ChatPromptTemplate。

    Returns:
        ChatPromptTemplate，含 system / chat_history / input / agent_scratchpad
    """
    system_prompt = load_prompt("system.txt")

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])


def build_agent() -> Tuple[AgentExecutor, Any]:
    """构建 Tool-Calling Agent：知识库 + 联网搜索 + 股价查询。

    Agent 会根据用户问题自动选择使用哪些工具，然后综合给出答案。

    Returns:
        (agent_executor, retriever) 元组
        - agent_executor: AgentExecutor 实例，调用 .invoke({"input": q, "chat_history": [...]})
        - retriever: MMR 检索器（供前端展示来源时使用）

    Raises:
        FileNotFoundError: 向量库不存在
        ValueError: API Key 未配置
    """
    # 校验 API Key
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY 未设置。\n"
            "免费获取：https://console.groq.com → 注册 → API Keys\n"
            "本地开发：写入 .env 文件 → GROQ_API_KEY=gsk_xxxxx\n"
            "Streamlit Cloud：Manage app → Secrets → 填入 TOML 格式密钥后 Reboot"
        )

    # 1. 加载向量库 & 创建检索器
    vectorstore = _load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RETRIEVER_K,
            "fetch_k": RETRIEVER_FETCH_K,
        },
    )

    # 将检索器引用存入模块变量，供 _retrieve_knowledge 使用
    _retriever_ref["retriever"] = retriever

    # 2. 创建 LLM（agent 用稍高温度以支持工具调用灵活性）
    llm = _create_llm(temperature=0.3)

    # 3. 创建工具列表
    tools = _create_agent_tools()

    # 4. 创建 Prompt
    prompt = _create_agent_prompt()

    # 5. 创建 Agent
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

    # 6. 创建 AgentExecutor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=6,
        handle_parsing_errors=True,
    )

    return agent_executor, retriever


# ============================================================
# 对话历史格式化
# ============================================================

def format_chat_history(history: list) -> list:
    """将 Streamlit session 消息列表转为 LangChain chat_history 格式。

    Streamlit 格式:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    LangChain Chain 格式（向后兼容）:
        [(human_msg_str, ai_msg_str), ...]
    LangChain Agent 格式:
        [HumanMessage(...), AIMessage(...), ...]

    Args:
        history: Streamlit 消息列表（含 role/content 的 dict）

    Returns:
        LangChain 对话历史元组列表（兼容 ConversationalRetrievalChain）
    """
    chat_history = []
    user_msg = None
    for msg in history:
        if msg["role"] == "user":
            user_msg = msg["content"]
        elif msg["role"] == "assistant" and user_msg is not None:
            chat_history.append((user_msg, msg["content"]))
            user_msg = None

    # 仅保留最近 N 轮
    return chat_history[-MAX_HISTORY_TURNS:]


def format_chat_history_for_agent(history: list) -> list:
    """将 Streamlit session 消息列表转为 Agent 可用的 BaseMessage 列表。

    Args:
        history: Streamlit 消息列表（含 role/content 的 dict）

    Returns:
        LangChain BaseMessage 列表 [HumanMessage, AIMessage, ...]
    """
    messages = []
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    # 仅保留最近 N*2 条消息（N 轮对话）
    limit = MAX_HISTORY_TURNS * 2
    return messages[-limit:]

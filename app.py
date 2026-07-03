"""
SpaceX 股票分析问答助手 — Streamlit 前端入口。

支持：
- RAG 知识库检索（本地文档）
- 联网搜索（DuckDuckGo）
- 实时股价查询（yfinance）

用法：
    streamlit run app.py
"""

import streamlit as st
import os

from rag_pipeline import build_agent, format_chat_history_for_agent
from ingest import ingest as run_ingest
from config import CHROMA_DB_PATH, DOCS_PATH, CHROMA_COLLECTION_NAME, ENABLE_WEB_SEARCH, ENABLE_STOCK_LOOKUP


# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="SpaceX 股票分析助手",
    page_icon="🚀",
    layout="wide",
)

st.title("🚀 SpaceX 股票分析问答助手")
st.caption("基于 RAG 检索增强生成 + 实时联网搜索 — 知识库与实时数据双驱动")


# ============================================================
# 会话状态初始化
# ============================================================
def init_session():
    """初始化 Streamlit session state。"""
    defaults = {
        "messages": [],               # 对话历史: [{"role": "user/assistant", "content": "...", "sources": [...], "tools_used": [...]}]
        "agent": None,                # AgentExecutor 实例
        "retriever": None,            # 检索器（供查看来源）
        "knowledge_ready": False,     # 知识库是否就绪
        "knowledge_error": "",        # 知识库错误信息
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def load_agent():
    """尝试加载 Agent，状态写入 session。"""
    if st.session_state.agent is not None:
        return  # 已加载

    try:
        agent, retriever = build_agent()
        st.session_state.agent = agent
        st.session_state.retriever = retriever
        st.session_state.knowledge_ready = True
        st.session_state.knowledge_error = ""
    except FileNotFoundError as e:
        st.session_state.knowledge_ready = False
        st.session_state.knowledge_error = str(e)
    except ValueError as e:
        st.session_state.knowledge_ready = False
        st.session_state.knowledge_error = str(e)
    except Exception as e:
        st.session_state.knowledge_ready = False
        st.session_state.knowledge_error = f"未知错误: {e}"


# ============================================================
# 工具使用情况展示
# ============================================================
TOOL_LABELS = {
    "retrieve_knowledge": ("📚", "知识库检索"),
    "search_web": ("🌐", "联网搜索"),
    "get_stock_price": ("📊", "股价查询"),
}

TOOL_ORDER = ["retrieve_knowledge", "search_web", "get_stock_price"]


def extract_tool_info(intermediate_steps: list) -> list:
    """从 Agent intermediate_steps 中提取工具使用信息。

    Args:
        intermediate_steps: AgentExecutor 返回的中间步骤
            [(AgentAction, output_str), ...]

    Returns:
        [{"tool": "retrieve_knowledge", "input": "...", "output": "..."}, ...]
    """
    info = []
    for step in intermediate_steps:
        action, output = step
        tool_name = action.tool
        tool_input = action.tool_input
        # tool_input 可能是 dict {"query": "..."} 或直接字符串
        if isinstance(tool_input, dict):
            query = tool_input.get("query", str(tool_input))
        else:
            query = str(tool_input)
        info.append({
            "tool": tool_name,
            "input": query,
            "output": str(output),
        })
    return info


def render_tool_badges(tools_used: list):
    """在回答上方渲染工具使用标签。

    Args:
        tools_used: extract_tool_info() 返回的列表
    """
    if not tools_used:
        return

    # 去重统计
    seen_tools = []
    for t in tools_used:
        if t["tool"] not in seen_tools:
            seen_tools.append(t["tool"])

    badges = []
    for tool_name in seen_tools:
        emoji, label = TOOL_LABELS.get(tool_name, ("🔧", tool_name))
        badges.append(f"{emoji} {label}")

    st.caption(" | ".join(badges) + " · 实时数据已整合")


def render_tool_sources(tools_used: list):
    """在折叠区域内展示工具调用的详细来源。

    Args:
        tools_used: extract_tool_info() 返回的列表
    """
    if not tools_used:
        return

    with st.expander("🔍 查看数据来源与工具调用详情"):
        for i, t in enumerate(tools_used, 1):
            emoji, label = TOOL_LABELS.get(t["tool"], ("🔧", t["tool"]))
            st.caption(f"**{i}. {emoji} {label}**")
            st.caption(f"查询：_{t['input']}_")

            # 截断过长输出
            output = t["output"]
            if len(output) > 800:
                output = output[:800] + "\n\n... (内容已截断)"
            st.text(output)
            st.divider()


# ============================================================
# 侧边栏
# ============================================================
def render_sidebar():
    """渲染侧边栏：自动初始化知识库、工具状态、操作按钮。"""
    with st.sidebar:
        st.header("📋 系统状态")

        # --- 自动初始化知识库 ---
        # 如果 docs/ 下有文档但向量库未初始化（或为空），自动执行 ingestion
        needs_ingest = False
        if os.path.isdir(DOCS_PATH):
            doc_files = [
                f for f in os.listdir(DOCS_PATH)
                if os.path.isfile(os.path.join(DOCS_PATH, f))
                and os.path.splitext(f)[1].lower() in {".pdf", ".txt", ".md"}
            ]
            if doc_files:
                if not os.path.isdir(CHROMA_DB_PATH):
                    needs_ingest = True
                else:
                    # 检查向量库是否为空（仅占位文档）
                    try:
                        from langchain_chroma import Chroma
                        from config import get_embeddings
                        emb = get_embeddings()
                        db = Chroma(
                            collection_name=CHROMA_COLLECTION_NAME,
                            embedding_function=emb,
                            persist_directory=CHROMA_DB_PATH,
                        )
                        if db._collection.count() <= 1:
                            needs_ingest = True
                    except Exception:
                        needs_ingest = True

        if needs_ingest:
            with st.spinner("🔄 正在自动初始化知识库（首次运行）..."):
                try:
                    run_ingest(reset=True)
                    st.success("✅ 知识库初始化完成")
                except Exception as e:
                    st.warning(f"⚠️ 自动初始化失败: {e}")

        load_agent()

        # 向量库路径
        if os.path.isdir(CHROMA_DB_PATH):
            try:
                from langchain_chroma import Chroma
                from config import get_embeddings
                emb = get_embeddings()
                db = Chroma(
                    collection_name=CHROMA_COLLECTION_NAME,
                    embedding_function=emb,
                    persist_directory=CHROMA_DB_PATH,
                )
                count = db._collection.count()
                st.success(f"✅ 知识库就绪（{count} 条向量）")
            except Exception:
                st.success("✅ Chroma 向量库已就绪")
        else:
            st.info("ℹ️ 知识库为空，将依赖联网搜索")

        # 文档目录
        doc_files = []
        if os.path.isdir(DOCS_PATH):
            doc_files = [
                f for f in os.listdir(DOCS_PATH)
                if os.path.isfile(os.path.join(DOCS_PATH, f))
                and os.path.splitext(f)[1].lower() in {".pdf", ".txt", ".md"}
            ]
        st.metric("知识库文档数", len(doc_files))
        if doc_files:
            with st.expander("查看文档列表"):
                for f in doc_files:
                    st.write(f"📄 {f}")

        # 工具状态
        st.divider()
        st.header("🔧 实时数据工具")
        web_status = "✅ 已启用" if ENABLE_WEB_SEARCH else "❌ 已禁用"
        stock_status = "✅ 已启用" if ENABLE_STOCK_LOOKUP else "❌ 已禁用"
        st.caption(f"🌐 联网搜索: {web_status}")
        st.caption(f"📊 股价查询: {stock_status}")

        st.divider()

        # 操作按钮
        st.header("⚙️ 操作")
        if st.button("🔄 重新加载系统", use_container_width=True):
            st.session_state.agent = None
            st.session_state.retriever = None
            st.session_state.knowledge_ready = False
            st.cache_resource.clear()
            st.rerun()

        if st.button("🗑️ 清除对话历史", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()

        st.caption(
            "⚠️ 以上内容仅基于现有资料整理，不构成任何投资建议。"
            "投资决策请结合专业财务顾问意见，自行承担风险。"
        )


# ============================================================
# 主对话区域
# ============================================================
def render_chat():
    """渲染对话界面与消息。"""
    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            # 如果是 assistant 消息且有工具使用，先显示工具标签
            if msg["role"] == "assistant" and msg.get("tools_used"):
                render_tool_badges(msg["tools_used"])

            st.markdown(msg["content"])

            # 展示来源
            if msg["role"] == "assistant":
                # 工具详细来源
                if msg.get("tools_used"):
                    render_tool_sources(msg["tools_used"])

                # 知识库来源（向后兼容）
                if msg.get("sources"):
                    with st.expander("📎 参考来源（知识库）"):
                        for src in msg["sources"]:
                            st.caption(
                                f"**{src.get('source', '未知文档')}** "
                                f"（第 {src.get('page', '?')} 页）"
                            )
                            st.write(src.get("content", ""))

    # 输入框
    if prompt := st.chat_input("请输入你的 SpaceX 相关问题（支持联网搜索与股价查询）..."):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 生成回答
        with st.chat_message("assistant"):
            if not st.session_state.knowledge_ready:
                err = st.session_state.knowledge_error or "知识库为空，请先运行 ingest.py 摄取文档。"
                st.error(err)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"抱歉，系统未就绪：{err}",
                    "sources": [],
                    "tools_used": [],
                })
                return

            with st.spinner("正在分析中（可能需要联网搜索或查询股价）..."):
                try:
                    agent = st.session_state.agent

                    # 转换对话历史（不含当前问题）
                    chat_history = format_chat_history_for_agent(
                        st.session_state.messages[:-1]
                    )

                    # 调用 Agent
                    result = agent.invoke({
                        "input": prompt,
                        "chat_history": chat_history,
                    })

                    answer = result.get("output", "")
                    intermediate_steps = result.get("intermediate_steps", [])

                    # 提取工具使用信息
                    tools_used = extract_tool_info(intermediate_steps)

                    # 显示工具标签
                    if tools_used:
                        render_tool_badges(tools_used)

                    # 展示回答
                    st.markdown(answer)

                    # 展示来源
                    if tools_used:
                        render_tool_sources(tools_used)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": [],
                        "tools_used": tools_used,
                    })

                except Exception as e:
                    error_msg = f"请求失败: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"抱歉，处理请求时出错：{e}",
                        "sources": [],
                        "tools_used": [],
                    })


# ============================================================
# 主入口
# ============================================================
def main():
    init_session()
    render_sidebar()

    # 主区域：欢迎词（仅首次）
    if not st.session_state.messages:
        st.info(
            "👋 欢迎使用 SpaceX 股票分析助手！\n\n"
            "我可以帮你：\n"
            "- 📚 基于本地知识库分析 SpaceX 商业模式、发射数据、竞争对手\n"
            "- 🌐 **联网搜索**最新估值、融资、发射进展\n"
            "- 📊 **实时查询**相关上市公司股价（RKLB、LUNR、BA、LMT 等）\n\n"
            "> 💡 **提示**：直接输入问题即可，我会自动决定使用知识库还是联网搜索。\n"
            "> 例如：「SpaceX 最新估值多少？」「Rocket Lab 今天股价？」「对比 SpaceX 和 Blue Origin」"
        )

    render_chat()


if __name__ == "__main__":
    main()

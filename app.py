"""
SpaceX 股票分析问答助手 —— Streamlit 前端入口。

用法：
    streamlit run app.py
"""

import os
import streamlit as st

# ============================================================
# 将 Streamlit Cloud Secrets 注入到环境变量
# DeepSeek / LangChain 等库通过 os.environ 读取 API Key，
# 而 Streamlit Cloud 的 secrets 只存在于 st.secrets，需要手动注入
# ============================================================
try:
    for key, value in st.secrets.items():
        os.environ[key] = str(value)
except Exception:
    pass  # 本地开发时 st.secrets 可能不存在

# ============================================================
# 登录认证
# ============================================================
from utils.config_handler import access_password

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = not bool(access_password)

if not st.session_state["authenticated"]:
    st.title("🚀 SpaceX 股票分析问答助手")
    st.divider()
    password_input = st.text_input("请输入访问密码", type="password")

    if password_input:
        if password_input == access_password:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("密码错误，请重试")
    st.stop()

# ============================================================
# 主应用
# ============================================================
from agent.tools.react_agent import ReactAgent
from rag.vector_store import VectorStoreService
from utils.config_handler import chroma_config
from utils.path_tool import get_abs_path

st.set_page_config(page_title="SpaceX 股票分析助手", page_icon="🚀", layout="wide")
st.title("🚀 SpaceX 股票分析问答助手")
st.caption("基于 RAG 检索增强生成 + Tool-Calling Agent —— 知识库与实时联网数据双驱动")

TOOL_LABELS = {
    "retrieve_knowledge": ("📚", "知识库检索"),
    "search_web_online": ("🌐", "联网搜索"),
    "get_stock_price_info": ("📊", "股价查询"),
    "get_spacex_latest_news": ("🚀", "SpaceX最新资讯"),
    "get_related_stocks": ("📈", "同赛道股价一览"),
}


def init_session():
    """初始化 Streamlit session state"""
    if "agent" not in st.session_state:
        with st.spinner("正在初始化 Agent..."):
            st.session_state["agent"] = ReactAgent()
    if "messages" not in st.session_state:
        st.session_state["messages"] = []


def get_kb_doc_count() -> int:
    """获取向量库中已写入的文档片段数量，用于侧边栏展示知识库状态"""
    try:
        vector_store = VectorStoreService().vector_store
        return vector_store._collection.count()
    except Exception:
        return 0


def render_sidebar():
    """侧边栏：知识库状态、清除对话、重新加载文档"""
    with st.sidebar:
        st.header("📋 系统状态")

        docs_dir = get_abs_path(chroma_config["data_path"])

        # 首次运行/云端冷启动时向量库为空，但 docs/ 下有文档，则自动摄取一次
        if get_kb_doc_count() == 0 and os.path.isdir(docs_dir):
            has_docs = any(
                os.path.splitext(f)[1].lower() in {".pdf", ".txt", ".md"}
                for f in os.listdir(docs_dir)
                if os.path.isfile(os.path.join(docs_dir, f))
            )
            if has_docs:
                with st.spinner("🔄 正在自动初始化知识库（首次运行）..."):
                    try:
                        VectorStoreService().load_document()
                    except Exception as e:
                        st.warning(f"⚠️ 知识库自动初始化失败: {e}")

        doc_count = get_kb_doc_count()
        if doc_count > 0:
            st.success(f"✅ 知识库就绪（{doc_count} 条向量）")
        else:
            st.info("ℹ️ 知识库为空，将依赖联网搜索工具")
        doc_files = []
        if os.path.isdir(docs_dir):
            doc_files = [
                f for f in os.listdir(docs_dir)
                if os.path.isfile(os.path.join(docs_dir, f))
                and os.path.splitext(f)[1].lower() in {".pdf", ".txt", ".md"}
            ]
        st.metric("知识库文档数", len(doc_files))
        if doc_files:
            with st.expander("查看文档列表"):
                for f in doc_files:
                    st.write(f"📄 {f}")

        st.divider()
        st.header("⚙️ 操作")

        if st.button("🔄 重新加载文档", use_container_width=True):
            with st.spinner("正在摄取 docs/ 目录下的文档..."):
                try:
                    VectorStoreService().load_document()
                    st.success("✅ 文档摄取完成")
                except Exception as e:
                    st.warning(f"⚠️ 摄取失败: {e}")
            st.rerun()

        if st.button("🗑️ 清除对话", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

        st.divider()
        st.caption(
            "⚠️ 以上内容仅基于现有资料整理，不构成任何投资建议。"
            "投资决策请结合专业财务顾问意见，自行承担风险。"
        )


def render_tool_badges(tool_calls: list):
    """在回答上方渲染工具使用标签"""
    if not tool_calls:
        return
    seen = []
    for t in tool_calls:
        if t["tool"] not in seen:
            seen.append(t["tool"])
    badges = [f"{TOOL_LABELS.get(name, ('🔧', name))[0]} {TOOL_LABELS.get(name, ('🔧', name))[1]}" for name in seen]
    st.caption(" | ".join(badges))


def render_tool_sources(tool_calls: list):
    """折叠展示工具调用的详细来源"""
    if not tool_calls:
        return
    with st.expander("📎 参考来源（工具调用详情）"):
        for i, t in enumerate(tool_calls, 1):
            emoji, label = TOOL_LABELS.get(t["tool"], ("🔧", t["tool"]))
            st.caption(f"**{i}. {emoji} {label}**")
            output = t["output"]
            if len(output) > 800:
                output = output[:800] + "\n\n... (内容已截断)"
            st.text(output)
            st.divider()


def render_chat():
    """渲染对话界面与消息"""
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("tool_info"):
                render_tool_badges(msg["tool_info"])
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("tool_info"):
                render_tool_sources(msg["tool_info"])

    prompt = st.chat_input("请输入你的 SpaceX 相关问题（支持联网搜索与股价查询）...")

    if prompt:
        st.chat_message("user").write(prompt)
        st.session_state["messages"].append({"role": "user", "content": prompt})

        # 只保留 role/content 传给 agent，避免自定义字段（如 tool_info）
        # 与 LangChain 消息里的保留字段（如 tool_calls）同名冲突
        chat_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state["messages"][:-1]
        ]

        response_chunks = []
        with st.spinner("正在分析中（可能需要检索知识库、联网搜索或查询股价）..."):
            agent = st.session_state["agent"]
            res_stream = agent.execute_stream(prompt, chat_history)

            def capture(generator, cache_list):
                for chunk in generator:
                    cache_list.append(chunk)
                    yield chunk

            with st.chat_message("assistant"):
                st.write_stream(capture(res_stream, response_chunks))
                tool_calls = getattr(agent, "last_tool_calls", [])
                if tool_calls:
                    render_tool_sources(tool_calls)

        answer = response_chunks[-1] if response_chunks else "抱歉，未能生成回答。"
        st.session_state["messages"].append({
            "role": "assistant",
            "content": answer,
            "tool_info": getattr(st.session_state["agent"], "last_tool_calls", []),
        })


def main():
    init_session()
    render_sidebar()

    if not st.session_state["messages"]:
        st.info(
            "👋 欢迎使用 SpaceX 股票分析助手！\n\n"
            "我可以帮你：\n"
            "- 📚 基于本地知识库分析 SpaceX 商业模式、发射数据、竞争对手\n"
            "- 🌐 **联网搜索**最新估值、融资、发射进展\n"
            "- 📊 **实时查询**相关上市公司股价（RKLB、LUNR、BA、LMT 等）\n\n"
            "> 💡 直接输入问题即可，我会自动决定调用哪个工具。"
        )

    render_chat()


if __name__ == "__main__":
    main()

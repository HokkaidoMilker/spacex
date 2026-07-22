from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage

from model.factory import chat_model
from utils.prompt_loader import load_system_prompt
from utils.config_handler import agent_config
from agent.tools.agent_tools import (
    retrieve_knowledge,
    search_web_online,
    get_stock_price_info,
    get_spacex_latest_news,
    get_related_stocks,
)
from agent.tools.middleware import monitor_tool, log_before_model


def _build_tools() -> list:
    """根据 config/agent.yml 中的开关组装工具列表"""
    tools = [retrieve_knowledge]
    if agent_config.get("enable_web_search", True):
        tools += [search_web_online, get_spacex_latest_news]
    if agent_config.get("enable_stock_lookup", True):
        tools += [get_stock_price_info, get_related_stocks]
    return tools


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompt(),
            tools=_build_tools(),
            middleware=[monitor_tool, log_before_model],
        )
        # 最近一次问答中被调用的工具及其输出，供前端展示「参考来源」
        self.last_tool_calls: list[dict] = []

    def execute_stream(self, query: str, chat_history: list[dict] | None = None):
        """流式执行一次问答，chat_history 为 [{"role": "user/assistant", "content": "..."}, ...]"""
        messages = list(chat_history or [])
        messages.append({"role": "user", "content": query})
        input_dict = {"messages": messages}

        self.last_tool_calls = []
        seen_tool_call_ids: set[str] = set()

        for chunk in self.agent.stream(input_dict, stream_mode="values"):
            state_messages = chunk["messages"]

            for msg in state_messages:
                if isinstance(msg, ToolMessage) and msg.tool_call_id not in seen_tool_call_ids:
                    seen_tool_call_ids.add(msg.tool_call_id)
                    self.last_tool_calls.append({
                        "tool": msg.name,
                        "output": str(msg.content),
                    })

            latest_message = state_messages[-1]
            if isinstance(latest_message, AIMessage) and latest_message.content:
                yield latest_message.content.strip() + "\n"


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("SpaceX最新的估值是多少？和Rocket Lab相比竞争优势在哪？"):
        print(chunk, end="", flush=True)
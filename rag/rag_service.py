from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_qa_chain_prompt
from model.factory import chat_model


class RagSummerizeService:
    def __init__(self):
        self.vectorstore = VectorStoreService()
        self.retriever = self.vectorstore.retriever()
        self.prompt_txt = load_qa_chain_prompt()
        self.prompt_template = PromptTemplate.from_template(self.prompt_txt)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        return self.prompt_template | self.model | StrOutputParser()

    def retriever_docs(self, query: str) -> list[Document]:
        """用检索器返回查询内容"""
        return self.retriever.invoke(query)

    def rag_summerize(self, query: str) -> str:
        """检索知识库并结合 qa_chain 提示词合成结构化回答，供 Agent 工具调用"""
        context_docs = self.retriever_docs(query)
        if not context_docs:
            return "当前知识库中未找到相关内容，建议查阅最新资料"

        context = ""
        for counter, doc in enumerate(context_docs, start=1):
            source = doc.metadata.get("source", "未知来源")
            context += f"参考资料{counter}（来源：{source}）：{doc.page_content}\n"

        return self.chain.invoke({
            "context": context,
            "chat_history": "",
            "question": query,
        })


if __name__ == "__main__":
    rag = RagSummerizeService()
    print(rag.rag_summerize("SpaceX的核心收入来源是什么"))
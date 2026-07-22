from abc import ABC, abstractmethod

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings

from utils.config_handler import rag_config, dashscope_api_key, dashscope_base_url


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Embeddings | BaseChatModel:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Embeddings | BaseChatModel:
        return ChatOpenAI(
            model=rag_config["chat_model_name"],
            api_key=dashscope_api_key,
            base_url=dashscope_base_url,
            temperature=rag_config.get("temperature", 0.3),
        )


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Embeddings | BaseChatModel:
        return DashScopeEmbeddings(
            model=rag_config["embedding_model_name"],
            dashscope_api_key=dashscope_api_key,
        )


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()

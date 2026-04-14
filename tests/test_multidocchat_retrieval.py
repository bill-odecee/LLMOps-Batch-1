import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

# Patch required langchain imports before importing the module under test.
langchain_core = types.ModuleType("langchain_core")
langchain_core.messages = types.ModuleType("langchain_core.messages")
langchain_core.output_parsers = types.ModuleType("langchain_core.output_parsers")
langchain_core.prompts = types.ModuleType("langchain_core.prompts")

langchain_core.messages.BaseMessage = object


class FakeStrOutputParser:
    def __ror__(self, other):
        return self

    def __or__(self, other):
        return self

    def invoke(self, payload):
        return "final answer"


langchain_core.output_parsers.StrOutputParser = FakeStrOutputParser


class FakeChatPromptTemplate:
    @classmethod
    def from_template(cls, template):
        return cls()

    @classmethod
    def from_messages(cls, messages):
        return cls()

    def __ror__(self, other):
        return self

    def __or__(self, other):
        return other


langchain_core.prompts.ChatPromptTemplate = FakeChatPromptTemplate
langchain_core.prompts.MessagesPlaceholder = lambda name: None

sys.modules["langchain_core"] = langchain_core
sys.modules["langchain_core.messages"] = langchain_core.messages
sys.modules["langchain_core.output_parsers"] = langchain_core.output_parsers
sys.modules["langchain_core.prompts"] = langchain_core.prompts

vectorstores = types.ModuleType("langchain_community.vectorstores")


class FakeFAISS:
    @classmethod
    def load_local(cls, index_path, embeddings, index_name="index"):
        return SimpleNamespace(
            as_retriever=lambda search_type, search_kwargs: FakeRetriever()
        )


vectorstores.FAISS = FakeFAISS
sys.modules.setdefault("langchain_community", types.ModuleType("langchain_community"))
sys.modules["langchain_community.vectorstores"] = vectorstores

# Patch utils.model_loader before importing the module under test.
utils_model_loader = types.ModuleType("utils.model_loader")


class FakeLLM:
    def __ror__(self, other):
        return self

    def __or__(self, other):
        return other

    def invoke(self, payload):
        return "llm answer"


class FakeModelLoader:
    def load_llm(self):
        return FakeLLM()

    def load_embeddings(self):
        return object()


utils_model_loader.Modelloader = FakeModelLoader
sys.modules["utils.model_loader"] = utils_model_loader

# Now import the module under test.
retrieval = importlib.import_module("src.multidocchat.retrieval")


class FakePrompt:
    def __ror__(self, other):
        return self

    def __or__(self, other):
        return other


class FakeRetriever:
    def __ror__(self, other):
        return self

    def __or__(self, other):
        return other


@pytest.fixture(autouse=True)
def fake_prompt_registry(monkeypatch):
    monkeypatch.setitem(
        retrieval.PROMPT_REGISTRY,
        retrieval.PromptType.CONTEXTUALIZE_QUESTION.value,
        FakePrompt(),
    )
    monkeypatch.setitem(
        retrieval.PROMPT_REGISTRY, retrieval.PromptType.CONTEXT_QA.value, FakePrompt()
    )
    yield


def test_format_docs_concatenates_page_content():
    docs = [SimpleNamespace(page_content="a"), SimpleNamespace(page_content="b"), "c"]

    result = retrieval.ConversationalRAG._format_docs(docs)

    assert result == "a\n\nb\n\nc"


def test_init_with_retriever_builds_chain():
    retriever_obj = FakeRetriever()
    rag = retrieval.ConversationalRAG(session_id="session1", retriever=retriever_obj)

    assert rag.chain is not None
    assert rag.retriever is retriever_obj


def test_init_without_retriever_raises():
    with pytest.raises(
        retrieval.DocumentPortalException,
        match="Failed to initialize ConversationalRAG",
    ):
        retrieval.ConversationalRAG(session_id="session2", retriever=None)


def test_load_retriever_from_faiss_loads_retriever_and_builds_chain(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(retrieval, "Modelloader", FakeModelLoader)
    fake_faiss = FakeFAISS()
    monkeypatch.setattr(retrieval, "FAISS", FakeFAISS)

    os_dir = tmp_path / "index"
    os_dir.mkdir()

    rag = retrieval.ConversationalRAG(session_id="session3", retriever=FakeRetriever())
    retriever_obj = rag.load_retriever_from_faiss(str(os_dir), k=2, index_name="index")

    assert retriever_obj is not None
    assert rag.chain is not None
    assert rag.retriever is not None


def test_load_retriever_from_faiss_missing_directory_raises(monkeypatch):
    monkeypatch.setattr(retrieval, "Modelloader", FakeModelLoader)
    monkeypatch.setattr(retrieval.os.path, "isdir", lambda path: False)

    rag = retrieval.ConversationalRAG(session_id="session4", retriever=FakeRetriever())

    with pytest.raises(
        retrieval.DocumentPortalException, match="FAISS index directory not found"
    ):
        rag.load_retriever_from_faiss("missing_path")


def test_invoke_returns_answer_when_chain_ready():
    retriever_obj = FakeRetriever()
    rag = retrieval.ConversationalRAG(session_id="session5", retriever=retriever_obj)
    rag.chain = SimpleNamespace(invoke=lambda payload: "it works")

    answer = rag.invoke("question", chat_history=[])

    assert answer == "it works"


def test_invoke_raises_when_no_chain():
    retriever_obj = FakeRetriever()
    rag = retrieval.ConversationalRAG(session_id="session6", retriever=retriever_obj)
    rag.chain = None

    with pytest.raises(
        retrieval.DocumentPortalException, match="RAG chain not initialized"
    ):
        rag.invoke("question")

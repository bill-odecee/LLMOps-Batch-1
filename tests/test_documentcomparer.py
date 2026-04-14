import importlib
import sys
import types
from types import SimpleNamespace

import pytest

# Ensure utils.model_loader can import without needing langchain_google_genai installed.
fake_langchain_google_genai = types.ModuleType("langchain_google_genai")
fake_langchain_google_genai.GoogleGenerativeAIEmbeddings = lambda *args, **kwargs: None
fake_langchain_google_genai.ChatGoogleGenerativeAI = lambda *args, **kwargs: None
sys.modules["langchain_google_genai"] = fake_langchain_google_genai


def make_fake_prompt():
    class FakeChain:
        def __init__(self):
            self.invoked_inputs = None

        def __or__(self, other):
            return self

        def invoke(self, inputs):
            self.invoked_inputs = inputs
            return [
                {"Page": "1", "Changes": "No changes"},
                {"Page": "2", "Changes": "Updated summary"},
            ]

    class FakePrompt:
        def __or__(self, other):
            return FakeChain()

    return FakePrompt(), FakeChain


class FakeParser:
    def __init__(self, *args, **kwargs):
        pass

    def get_format_instructions(self):
        return "format instructions"


class FakeFixingParser:
    pass


class FakeLLM:
    pass


class FakeModelLoader:
    def __init__(self):
        pass

    def load_llm(self):
        return FakeLLM()


@pytest.fixture(autouse=True)
def patch_document_comparer(monkeypatch):
    data_module = importlib.import_module("src.doccompare.documentcomparer")

    fake_prompt, _ = make_fake_prompt()

    monkeypatch.setattr(data_module, "load_dotenv", lambda override=True: None)
    monkeypatch.setattr(data_module, "Modelloader", FakeModelLoader)
    monkeypatch.setattr(data_module, "JsonOutputParser", FakeParser)
    monkeypatch.setattr(
        data_module,
        "OutputFixingParser",
        SimpleNamespace(from_llm=lambda parser, llm: FakeFixingParser()),
    )
    monkeypatch.setitem(data_module.PROMPT_REGISTRY, "document_comparison", fake_prompt)

    yield


def test_compare_documents_returns_dataframe(monkeypatch):
    data_module = importlib.import_module("src.doccompare.documentcomparer")

    comparer = data_module.DocumentComparerLLM()

    df = comparer.compare_documents("combined document content")

    assert not df.empty
    assert list(df.columns) == ["Page", "Changes"]
    assert df.iloc[0]["Page"] == "1"
    assert df.iloc[1]["Changes"] == "Updated summary"


def test_compare_documents_wraps_chain_errors(monkeypatch):
    data_module = importlib.import_module("src.doccompare.documentcomparer")

    class BrokenChain:
        def __or__(self, other):
            return self

        def invoke(self, inputs):
            raise RuntimeError("chain failure")

    class BrokenPrompt:
        def __or__(self, other):
            return BrokenChain()

    monkeypatch.setitem(
        data_module.PROMPT_REGISTRY, "document_comparison", BrokenPrompt()
    )

    comparer = data_module.DocumentComparerLLM()

    with pytest.raises(
        data_module.DocumentPortalException, match="Error in document comparison"
    ):
        comparer.compare_documents("combined document content")


def test_format_reponse_returns_dataframe_for_valid_response():
    data_module = importlib.import_module("src.doccompare.documentcomparer")
    comparer = data_module.DocumentComparerLLM()

    response = [{"Page": "1", "Changes": "No changes"}]
    df = comparer._format_reponse(response)

    assert list(df.columns) == ["Page", "Changes"]
    assert df.iloc[0]["Changes"] == "No changes"

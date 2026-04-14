import importlib
import sys
import types
from types import SimpleNamespace

import pytest

fake_langchain_google_genai = types.ModuleType("langchain_google_genai")
fake_langchain_google_genai.GoogleGenerativeAIEmbeddings = lambda *args, **kwargs: None
fake_langchain_google_genai.ChatGoogleGenerativeAI = lambda *args, **kwargs: None
sys.modules["langchain_google_genai"] = fake_langchain_google_genai


def make_fake_prompt():
    class FakeChain:
        def __init__(self):
            self.invoked_payload = None

        def __or__(self, other):
            return self

        def invoke(self, payload):
            self.invoked_payload = payload
            return {
                "Summary": ["Test summary"],
                "Title": "Test title",
                "Author": ["Tester"],
                "DateCreated": "2026-04-13",
                "LastModifiedDate": "2026-04-13",
                "Publisher": "Test publisher",
                "Language": "en",
                "PageCount": 1,
                "SentimentTone": "Neutral",
            }

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
def patch_data_analysis(monkeypatch):
    data_analysis = importlib.import_module("src.docanalyzer.data_analysis")

    fake_prompt, fake_chain_cls = make_fake_prompt()
    fake_output_fixing = FakeFixingParser()

    monkeypatch.setattr(data_analysis, "Modelloader", FakeModelLoader)
    monkeypatch.setattr(data_analysis, "JsonOutputParser", FakeParser)
    monkeypatch.setattr(
        data_analysis,
        "OutputFixingParser",
        SimpleNamespace(from_llm=lambda parser, llm: fake_output_fixing),
    )
    monkeypatch.setitem(data_analysis.PROMPT_REGISTRY, "document_analysis", fake_prompt)

    yield


def test_analyze_document_returns_expected_metadata(monkeypatch):
    data_analysis = importlib.import_module("src.docanalyzer.data_analysis")
    analyzer = data_analysis.DocumentAnalyzer()

    result = analyzer.analyze_document("test document text")

    assert result["Title"] == "Test title"
    assert result["Language"] == "en"
    assert result["PageCount"] == 1


def test_analyze_document_wraps_exceptions(monkeypatch):
    data_analysis = importlib.import_module("src.docanalyzer.data_analysis")

    class BrokenChain:
        def __or__(self, other):
            return self

        def invoke(self, payload):
            raise RuntimeError("chain failure")

    class BrokenPrompt:
        def __or__(self, other):
            return BrokenChain()

    monkeypatch.setitem(
        data_analysis.PROMPT_REGISTRY, "document_analysis", BrokenPrompt()
    )

    analyzer = data_analysis.DocumentAnalyzer()

    with pytest.raises(
        data_analysis.DocumentPortalException, match="Error during document analysis"
    ):
        analyzer.analyze_document("test document text")


def test_document_analyzer_init_wraps_modelloader_errors(monkeypatch):
    data_analysis = importlib.import_module("src.docanalyzer.data_analysis")

    class FailingLoader:
        def __init__(self):
            raise ValueError("loader failed")

    monkeypatch.setattr(data_analysis, "Modelloader", FailingLoader)

    with pytest.raises(
        data_analysis.DocumentPortalException,
        match="Initialization error in DocumentAnalyzer",
    ):
        data_analysis.DocumentAnalyzer()

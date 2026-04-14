import importlib
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

# Provide minimal fake modules only when missing.
sys.modules.setdefault("langchain_core", types.ModuleType("langchain_core"))
sys.modules.setdefault(
    "langchain_core.documents", types.ModuleType("langchain_core.documents")
)
sys.modules.setdefault(
    "langchain_text_splitters", types.ModuleType("langchain_text_splitters")
)
sys.modules.setdefault("langchain_community", types.ModuleType("langchain_community"))
sys.modules.setdefault(
    "langchain_community.vectorstores",
    types.ModuleType("langchain_community.vectorstores"),
)
sys.modules.setdefault(
    "langchain_google_genai", types.ModuleType("langchain_google_genai")
)
sys.modules.setdefault("fitz", types.ModuleType("fitz"))

if not hasattr(sys.modules["langchain_core.documents"], "Document"):
    sys.modules["langchain_core.documents"].Document = object

if not hasattr(
    sys.modules["langchain_text_splitters"], "RecursiveCharacterTextSplitter"
):
    sys.modules["langchain_text_splitters"].RecursiveCharacterTextSplitter = (
        lambda chunk_size, chunk_overlap: SimpleNamespace(
            split_documents=lambda docs: docs
        )
    )

if not hasattr(sys.modules["langchain_community.vectorstores"], "FAISS"):

    class FakeFAISS:
        def __init__(self, *args, **kwargs):
            self.saved = False

        @classmethod
        def load_local(cls, path, embeddings, allow_dangerous_deserialization=True):
            return cls()

        @classmethod
        def from_texts(cls, texts, embedding, metadatas=None):
            inst = cls()
            inst.texts = texts
            inst.metadatas = metadatas
            return inst

        def save_local(self, path):
            self.saved = True

        def add_documents(self, docs):
            self.added_documents = docs

        def as_retriever(self, search_type, search_kwargs):
            return SimpleNamespace(search_type=search_type, search_kwargs=search_kwargs)

    sys.modules["langchain_community.vectorstores"].FAISS = FakeFAISS

if not hasattr(sys.modules["langchain_google_genai"], "GoogleGenerativeAIEmbeddings"):
    sys.modules["langchain_google_genai"].GoogleGenerativeAIEmbeddings = (
        lambda *args, **kwargs: object()
    )
    sys.modules["langchain_google_genai"].ChatGoogleGenerativeAI = (
        lambda *args, **kwargs: object()
    )

if not hasattr(sys.modules["fitz"], "open"):
    sys.modules["fitz"].open = lambda path: None

# Provide fake utils modules to avoid importing large/optional dependencies.
utils_model_loader = sys.modules.setdefault(
    "utils.model_loader", types.ModuleType("utils.model_loader")
)
utils_model_loader.Modelloader = type(
    "FakeModelLoader",
    (),
    {
        "__init__": lambda self: None,
        "load_embeddings": lambda self: object(),
        "load_llm": lambda self: object(),
    },
)

utils_document_ops = sys.modules.setdefault(
    "utils.document_ops", types.ModuleType("utils.document_ops")
)
utils_document_ops.load_documents = lambda paths: []
utils_document_ops.concat_for_analysis = lambda docs: ""

utils_file_io = sys.modules.setdefault(
    "utils.file_io", types.ModuleType("utils.file_io")
)
utils_file_io.generate_session_id = lambda prefix="session": f"{prefix}_test"
utils_file_io.save_uploaded_files = lambda uploaded_files, target_dir: []

# Import the module under test after ensuring minimal dependencies are available.
data_ingestion = importlib.import_module("src.document_ingestion.data_ingestion")


class FakeModelLoader:
    def __init__(self):
        pass

    def load_embeddings(self):
        return object()

    def load_llm(self):
        return object()


class FakeDoc:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class FakePage:
    def __init__(self, text):
        self._text = text

    def get_text(self):
        return self._text


class FakePDF:
    def __init__(self, pages, encrypted=False):
        self.page_count = len(pages)
        self._pages = pages
        self.is_encrypted = encrypted

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def load_page(self, page_num):
        return FakePage(self._pages[page_num])


@pytest.fixture(autouse=True)
def patch_document_ingestion(monkeypatch):
    # Patch the model loader used by FaissManager, DocumentComparer, and ChatIngestor.
    monkeypatch.setattr(data_ingestion, "Modelloader", FakeModelLoader)
    yield


def test_faissmanager_fingerprint_uses_source_and_row_id():
    key = data_ingestion.FaissManager._fingerprint(
        "text", {"source": "file.pdf", "row_id": 123}
    )
    assert key == "file.pdf::123"


def test_faissmanager_fingerprint_uses_source_without_row_id():
    key = data_ingestion.FaissManager._fingerprint("text", {"source": "file.pdf"})
    assert key == "file.pdf::"


def test_faissmanager_fingerprint_uses_hash_when_source_missing():
    key1 = data_ingestion.FaissManager._fingerprint("text-one", {})
    key2 = data_ingestion.FaissManager._fingerprint("text-two", {})
    assert key1 != key2
    assert len(key1) == 64


def test_faissmanager_load_or_create_raises_when_no_texts_and_index_missing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(data_ingestion.FaissManager, "_exists", lambda self: False)
    manager = data_ingestion.FaissManager(str(tmp_path), model_loader=FakeModelLoader())

    with pytest.raises(
        data_ingestion.DocumentPortalException,
        match="No existing FAISS index and no data to create one",
    ):
        manager.load_or_create()


def test_faissmanager_load_or_create_loads_existing_index(tmp_path, monkeypatch):
    monkeypatch.setattr(data_ingestion.FaissManager, "_exists", lambda self: True)

    loaded = object()
    monkeypatch.setattr(
        data_ingestion, "FAISS", sys.modules["langchain_community.vectorstores"].FAISS
    )
    manager = data_ingestion.FaissManager(str(tmp_path), model_loader=FakeModelLoader())
    vs = manager.load_or_create(texts=["a"], metadatas=[{}])

    assert vs is not None


def test_faissmanager_add_documents_idempotently(tmp_path, monkeypatch):
    manager = data_ingestion.FaissManager(str(tmp_path), model_loader=FakeModelLoader())
    fake_vs = SimpleNamespace(
        add_documents=lambda docs: None, save_local=lambda path: None
    )
    manager.vs = fake_vs
    manager._meta = {"rows": {}}

    docs = [FakeDoc("content", {"source": "file.pdf"})]
    added = manager.add_documents(docs)

    assert added == 1
    assert "file.pdf::" in manager._meta["rows"]


def test_doc_handler_save_pdf_and_invalid_extension(tmp_path):
    handler = data_ingestion.DocHandler(data_dir=str(tmp_path), session_id="sess123")

    class FakeFile:
        def __init__(self, name, content):
            self.name = name
            self._content = content

        def read(self):
            return self._content

    saved_path = handler.save_pdf(FakeFile("document.pdf", b"data"))
    assert Path(saved_path).exists()
    assert Path(saved_path).read_bytes() == b"data"

    with pytest.raises(
        data_ingestion.DocumentPortalException, match="Invalid file type"
    ):
        handler.save_pdf(FakeFile("document.txt", b"data"))


def test_doc_handler_read_pdf(tmp_path, monkeypatch):
    handler = data_ingestion.DocHandler(data_dir=str(tmp_path), session_id="sess123")
    pdf_path = Path(handler.session_path) / "test.pdf"
    pdf_path.write_bytes(b"dummy")

    monkeypatch.setattr(
        data_ingestion.fitz,
        "open",
        lambda path: FakePDF(["page1", "page2"]),
        raising=False,
    )

    text = handler.read_pdf(str(pdf_path))
    assert "Page 1" in text
    assert "page2" in text


def test_document_comparer_save_and_combine(tmp_path, monkeypatch):
    comparer = data_ingestion.DocumentComparer(
        base_dir=str(tmp_path), session_id="sess123"
    )

    class FakeFile:
        def __init__(self, name, content):
            self.name = name
            self._content = content

        def read(self):
            return self._content

    ref_path, act_path = comparer.save_uploaded_files(
        FakeFile("a.pdf", b"ref"), FakeFile("b.pdf", b"act")
    )
    assert ref_path.exists() and act_path.exists()

    monkeypatch.setattr(
        data_ingestion.DocumentComparer,
        "read_pdf",
        lambda self, path: f"CONTENT {path.name}",
    )
    combined = comparer.combine_documents()
    assert "Document: a.pdf" in combined
    assert "Document: b.pdf" in combined


def test_document_comparer_clean_old_sessions(tmp_path):
    base = tmp_path / "sessions"
    older = base / "20260101"
    newer = base / "20260201"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)

    comparer = data_ingestion.DocumentComparer(
        base_dir=str(base), session_id="20260201"
    )
    assert older.exists()
    comparer.clean_old_sessions(keep_latest=1)
    assert not older.exists()
    assert newer.exists()


def test_chatingestor_resolve_dir_without_session(tmp_path, monkeypatch):
    monkeypatch.setattr(data_ingestion, "Modelloader", FakeModelLoader)
    ingestor = data_ingestion.ChatIngestor(
        temp_base=str(tmp_path),
        faiss_base=str(tmp_path / "faiss"),
        use_session_dirs=False,
        session_id="sess123",
    )
    assert ingestor.temp_dir == Path(str(tmp_path))
    assert ingestor.faiss_dir == Path(str(tmp_path / "faiss"))


def test_chatingestor_built_retriver(monkeypatch, tmp_path):
    monkeypatch.setattr(
        data_ingestion,
        "save_uploaded_files",
        lambda uploaded_files, target_dir: [tmp_path / "doc.pdf"],
    )
    monkeypatch.setattr(
        data_ingestion,
        "load_documents",
        lambda paths: [FakeDoc("text", {"source": "doc.pdf"})],
    )
    monkeypatch.setattr(
        data_ingestion,
        "RecursiveCharacterTextSplitter",
        lambda chunk_size, chunk_overlap: SimpleNamespace(
            split_documents=lambda docs: docs
        ),
    )

    class FakeFM:
        def __init__(self, *args, **kwargs):
            self.loaded = False

        def load_or_create(self, texts=None, metadatas=None):
            self.loaded = True
            return self

        def add_documents(self, docs):
            return len(docs)

        def as_retriever(self, search_type, search_kwargs):
            return SimpleNamespace(search_type=search_type, search_kwargs=search_kwargs)

    monkeypatch.setattr(data_ingestion, "FaissManager", FakeFM)
    monkeypatch.setattr(data_ingestion, "Modelloader", FakeModelLoader)

    ingestor = data_ingestion.ChatIngestor(
        temp_base=str(tmp_path),
        faiss_base=str(tmp_path / "faiss"),
        use_session_dirs=True,
        session_id="sess123",
    )
    retriever = ingestor.built_retriver(
        [SimpleNamespace(name="doc.pdf", read=lambda: b"pdf")]
    )

    assert retriever.search_type == "similarity"
    assert retriever.search_kwargs == {"k": 5}

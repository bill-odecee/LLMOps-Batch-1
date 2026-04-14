from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace

import pytest

from utils import document_ops
from exceptions.custom_exception import DocumentPortalException


class FakeLoader:
    def __init__(self, path: str, **kwargs):
        self.path = path

    def load(self):
        return [
            SimpleNamespace(page_content=f"loaded:{self.path}", metadata={"source": self.path})
        ]


class ErrorLoader:
    def __init__(self, path: str, **kwargs):
        self.path = path

    def load(self):
        raise RuntimeError("load failed")


class HandlerWithReadPDF:
    def read_pdf(self, path: str) -> str:
        return f"read_pdf:{path}"


class HandlerWithReadOld:
    def read_(self, path: str) -> str:
        return f"read_:{path}"


def test_load_documents_uses_supported_loaders_and_skips_unsupported(monkeypatch):
    monkeypatch.setattr(document_ops, "PyPDFLoader", FakeLoader)
    monkeypatch.setattr(document_ops, "Docx2txtLoader", FakeLoader)
    monkeypatch.setattr(document_ops, "TextLoader", FakeLoader)

    paths = [
        Path("document.pdf"),
        Path("notes.docx"),
        Path("readme.txt"),
        Path("unsupported.md"),
    ]

    docs = document_ops.load_documents(paths)

    assert len(docs) == 3
    assert [doc.page_content for doc in docs] == [
        "loaded:document.pdf",
        "loaded:notes.docx",
        "loaded:readme.txt",
    ]
    assert [doc.metadata["source"] for doc in docs] == [
        "document.pdf",
        "notes.docx",
        "readme.txt",
    ]


def test_load_documents_raises_document_portal_exception_on_loader_error(monkeypatch):
    monkeypatch.setattr(document_ops, "TextLoader", ErrorLoader)
    with pytest.raises(DocumentPortalException) as exc_info:
        document_ops.load_documents([Path("broken.txt")])

    assert "Error loading documents" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_concat_for_analysis_uses_source_file_path_or_unknown():
    docs = [
        SimpleNamespace(page_content="first", metadata={"source": "a.pdf"}),
        SimpleNamespace(page_content="second", metadata={"file_path": "b.docx"}),
        SimpleNamespace(page_content="third", metadata={}),
    ]

    result = document_ops.concat_for_analysis(docs)

    assert "SOURCE: a.pdf" in result
    assert "SOURCE: b.docx" in result
    assert "SOURCE: unknown" in result
    assert "first" in result
    assert "second" in result
    assert "third" in result


def test_read_pdf_via_handler_prefers_read_pdf_method():
    handler = HandlerWithReadPDF()
    result = document_ops.read_pdf_via_handler(handler, "file.pdf")
    assert result == "read_pdf:file.pdf"


def test_read_pdf_via_handler_falls_back_to_read_old_method():
    handler = HandlerWithReadOld()
    result = document_ops.read_pdf_via_handler(handler, "file.pdf")
    assert result == "read_:file.pdf"


def test_read_pdf_via_handler_raises_for_missing_methods():
    handler = SimpleNamespace()
    with pytest.raises(RuntimeError, match="DocHandler has neither read_pdf nor read_ method"):
        document_ops.read_pdf_via_handler(handler, "file.pdf")

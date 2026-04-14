from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace

import pytest

from utils import file_io
from exceptions.custom_exception import DocumentPortalException


class FakeUploadedFile:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def read(self):
        return self._content


class FakeUploadedBuffer:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getbuffer(self):
        return self._content


class BrokenUploadedFile:
    def __init__(self, name: str):
        self.name = name

    def read(self):
        raise RuntimeError("cannot read")


def test_generate_session_id_has_prefix_and_unique_suffix():
    session_id = file_io.generate_session_id(prefix="test")

    assert session_id.startswith("test_")
    parts = session_id.split("_")
    assert len(parts) == 4
    assert parts[1].isdigit()
    assert parts[2].isdigit()
    assert len(parts[3]) == 8


def test_save_uploaded_files_writes_supported_files(tmp_path, monkeypatch):
    monkeypatch.setattr(file_io.uuid, "uuid4", lambda: SimpleNamespace(hex="deadbeef"))
    uploaded = [FakeUploadedFile("document.pdf", b"pdf-data")]

    saved_paths = file_io.save_uploaded_files(uploaded, tmp_path)

    assert len(saved_paths) == 1
    saved_path = saved_paths[0]
    assert saved_path.exists()
    assert saved_path.suffix == ".pdf"
    assert saved_path.read_bytes() == b"pdf-data"


def test_save_uploaded_files_skips_unsupported_extension(tmp_path, monkeypatch):
    uploaded = [FakeUploadedFile("archive.zip", b"zip-data")]

    saved_paths = file_io.save_uploaded_files(uploaded, tmp_path)

    assert saved_paths == []
    assert list(tmp_path.iterdir()) == []


def test_save_uploaded_files_uses_getbuffer_when_no_read(tmp_path, monkeypatch):
    monkeypatch.setattr(file_io.uuid, "uuid4", lambda: SimpleNamespace(hex="cafebabe"))
    uploaded = [FakeUploadedBuffer("notes.txt", b"hello")]

    saved_paths = file_io.save_uploaded_files(uploaded, tmp_path)

    assert len(saved_paths) == 1
    assert saved_paths[0].read_bytes() == b"hello"


def test_save_uploaded_files_raises_document_portal_exception_on_write_error(tmp_path):
    uploaded = [BrokenUploadedFile("broken.pdf")]

    with pytest.raises(DocumentPortalException) as exc_info:
        file_io.save_uploaded_files(uploaded, tmp_path)

    assert "Failed to save uploaded files" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)

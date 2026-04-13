from __future__ import annotations
import os
import sys
import json
import hashlib
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any

import PyMuPDF  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Assuming these remain in your project structure
from utils.model_loader import Modelloader
from logger import GLOBAL_LOGGER as log
from exceptions.custom_exception import DocumentPortalException
from utils.file_io import generate_session_id

SUPPORTED_EXTENSIONS = {".pdf"}


class FaissManager:
    """
    Manages the FAISS vector store with content-hash based deduplication.
    """

    def __init__(self, index_dir: str, model_loader: Optional[Modelloader] = None):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.meta_path = self.index_dir / "ingested_meta.json"
        self._meta: Dict[str, Any] = {"hashes": {}}

        # Load existing metadata if available
        if self.meta_path.exists():
            try:
                self._meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
                if "hashes" not in self._meta:
                    self._meta["hashes"] = {}
            except Exception:
                log.warning("Failed to load metadata, initializing fresh state")
                self._meta = {"hashes": {}}

        self.model_loader = model_loader or Modelloader()
        self.emb = self.model_loader.load_embeddings()
        self.vs: Optional[FAISS] = None

        # Initialize text splitter for high-quality RAG
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )

    def _exists(self) -> bool:
        return (self.index_dir / "index.faiss").exists() and (
            self.index_dir / "index.pkl"
        ).exists()

    def _get_content_hash(self, text: str) -> str:
        """Generate a unique fingerprint based on the text content."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def __save_state(self):
        """Atomic-like save for both FAISS and Metadata."""
        if self.vs:
            self.vs.save_local(str(self.index_dir))
            self.meta_path.write_text(
                json.dumps(self._meta, indent=2), encoding="utf-8"
            )

    def add_documents(self, docs: List[Document]) -> int:
        """Chunks documents and adds only new content to the vector store."""
        if self.vs is None:
            self.load_or_create()

        # 1. Break down documents into smaller chunks
        chunks = self.splitter.split_documents(docs)

        new_chunks: List[Document] = []
        for chunk in chunks:
            c_hash = self._get_content_hash(chunk.page_content)

            # 2. Check if this specific piece of text is already known
            if c_hash not in self._meta["hashes"]:
                chunk.metadata["content_hash"] = c_hash
                self._meta["hashes"][c_hash] = True
                new_chunks.append(chunk)

        if new_chunks:
            self.vs.add_documents(new_chunks)
            self.__save_state()
            log.info(f"Added {len(new_chunks)} new chunks to FAISS index")

        return len(new_chunks)

    def load_or_create(
        self, texts: Optional[List[str]] = None, metadatas: Optional[List[dict]] = None
    ):
        if self._exists():
            self.vs = FAISS.load_local(
                str(self.index_dir),
                embeddings=self.emb,
                allow_dangerous_deserialization=True,
            )
            return self.vs

        if not texts:
            # Create an empty index if no initial data
            self.vs = FAISS.from_texts(texts=["initialization"], embedding=self.emb)
            self.__save_state()
            return self.vs

        self.vs = FAISS.from_texts(
            texts=texts, embedding=self.emb, metadatas=metadatas or []
        )
        self.__save_state()
        return self.vs


class DocHandler:
    """
    Handles PDF persistence and extraction with built-in validation.
    """

    def __init__(
        self, data_dir: Optional[str] = None, session_id: Optional[str] = None
    ):
        base = data_dir or os.getenv(
            "DATA_STORAGE_PATH", os.path.join(os.getcwd(), "data")
        )
        self.session_id = session_id or generate_session_id("session")
        self.session_path = Path(base) / "document_analysis" / self.session_id
        self.session_path.mkdir(parents=True, exist_ok=True)
        log.info("DocHandler initialized", session_id=self.session_id)

    def save_pdf(self, uploaded_file) -> Path:
        """Saves an uploaded file object safely."""
        try:
            filename = os.path.basename(uploaded_file.name)
            if not filename.lower().endswith(".pdf"):
                raise ValueError("Only PDF files are supported.")

            save_path = self.session_path / filename

            content = (
                uploaded_file.read()
                if hasattr(uploaded_file, "read")
                else uploaded_file.getbuffer()
            )
            save_path.write_bytes(content)

            log.info("File saved", file=filename, path=str(save_path))
            return save_path
        except Exception as e:
            raise DocumentPortalException(f"Failed to save PDF: {str(e)}", e) from e

    def extract_documents(self, pdf_path: Path) -> List[Document]:
        """Reads PDF and returns a list of LangChain Document objects (one per page)."""
        documents = []
        try:
            with fitz.open(pdf_path) as doc:
                if doc.is_encrypted:
                    raise ValueError(f"PDF is encrypted: {pdf_path.name}")

                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text = page.get_text().strip()

                    if text:
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": pdf_path.name,
                                    "page": page_num + 1,
                                    "session_id": self.session_id,
                                },
                            )
                        )
            log.info("Extraction complete", file=pdf_path.name, pages=len(documents))
            return documents
        except Exception as e:
            log.error("Extraction failed", error=str(e))
            raise DocumentPortalException("Error reading PDF content", e) from e

    def clean_session(self):
        """Wipes the current session directory."""
        if self.session_path.exists():
            shutil.rmtree(self.session_path)
            log.info("Session cleaned", session_id=self.session_id)

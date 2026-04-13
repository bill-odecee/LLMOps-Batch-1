#----Standard Import---
import os
from typing import List, Optional, Dict, Any

#---FastAPI fraework imports---
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware  #MIddleware to handle Cross-Originrequests
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates # For rendering HTML templates
from pathlib import Path

#----Internal modules imports----
from src.document_ingestion.data_ingestion import (
    DocHandler,
    DocumentComparer,
    ChatIngestor,
)
from src.docanalyzer.data_analysis import DocumentAnalyzer
from src.doccompare.documentcomparer import DocumentComparerLLM
from src.multidocchat.retrieval import ConversationalRAG
from utils.document_ops import FastAPIFileAdapter, read_pdf_via_handler # Adapts UploadFile to expected interface
from logger import GLOBAL_LOGGER as log

#----Configuration via Environment variables ----
FAISS_BASE = os.getenv("FAISS_BASE", "faiss_index")
UPLOAD_BASE = os.getenv("UPLOAD_BASE", "data")
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")

#----FastAPI application instance----
app = FastAPI(title="Document Portal API", version="1.0") # Project root (one level above / api)
BASE_DIR = Path(__file__).resolve().parent.parent  # Project root (one level above /api)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static") # For serving static files like CSS, JS
templates = Jinja2Templates(directory=str(BASE_DIR / "templates")) # For rendering HTML templates

#----CORS middleware: allows the frontend (any origin) to call this API ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,  # In production, specify allowed origins
    allow_methods=["*"],
    allow_headers=["*"],
)

#----Get / :Serves the main HTML page (frontend)----
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    log.info("Serving UI homepage")
    resp = templates.TemplateResponse(name="index.html", request=request) # Render index.html via jinja2 templates
    resp.headers["Cache-Control"] = "no-store" # Ensure the page is not cached
    return resp

#----GET /heath : Simple readiness probe ----
@app.get("/health")
def health() -> Dict[str, str]:
    log.info("Health check passed")
    return {"status": "ok", "service": "document_portal"}


#----ANALYZE----
# --- POST /analyze : Upload a single PDF and get an LLM-generated analysis ---
@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)) -> Any:
    try:
        log.info(f"Received file for analysis: {file.filename}")
        dh = DocHandler()                                    # Create a document handler for this request
        saved_path = dh.save_pdf(FastAPIFileAdapter(file))   # Persist the uploaded file to disk
        text = read_pdf_via_handler(dh, saved_path)          # Extract raw text from the saved PDF
        analyzer = DocumentAnalyzer()                        # Instantiate the LLM-based analyzer
        result = analyzer.analyze_document(text)             # Run analysis and get structured results
        log.info("Document analysis complete.")
        return JSONResponse(content=result)                  # Return analysis as JSON
    except HTTPException:
        raise  # Re-raise FastAPI HTTP errors as-is
    except Exception as e:
        log.exception("Error during document analysis")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

#----COMPARE----
#----POST / compare : Upload two PDFs & get side by side comparison
@app.post("/compare")
async def compare_documents(reference: UploadFile = File(...), actual: UploadFile = File(...)) -> Any:
    try:
        log.info(f"Comparing files: {reference.filename} vs {actual.filename}")
        dc = DocumentComparer()                       # Create a comparator session
        ref_path, act_path = dc.save_uploaded_files(         # Save both files to disk under a session folder
            FastAPIFileAdapter(reference), FastAPIFileAdapter(actual)
        )
        _ = ref_path, act_path                               # Paths stored for reference (unused directly here)
        combined_text = dc.combine_documents()               # Merge both documents' text into a single string
        comp = DocumentComparerLLM()                       # Instantiate the LLM-based comparator
        df = comp.compare_documents(combined_text)           # Run comparison; returns a pandas DataFrame
        log.info("Document comparison completed.")
        return {"rows": df.to_dict(orient="records"), "session_id": dc.session_id}  # Return rows + session ID
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Comparison failed")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")
    

# ---------- CHAT: INDEX ----------
# --- POST /chat/index : Upload one or more PDFs, chunk them, and build a FAISS vector index for RAG ---
@app.post("/chat/index")
async def chat_build_index(
    files: List[UploadFile] = File(...),           # One or more PDF files to index
    session_id: Optional[str] = Form(None),        # Optional session ID; auto-generated if omitted
    use_session_dirs: bool = Form(True),           # Whether to isolate indexes in per-session directories
    chunk_size: int = Form(1000),                  # Number of characters per text chunk
    chunk_overlap: int = Form(200),                # Overlap between consecutive chunks (improves retrieval)
    k: int = Form(5),                              # Number of top-k results the retriever will return
) -> Any:
    try:
        log.info(f"Indexing chat session. Session ID: {session_id}, Files: {[f.filename for f in files]}")
        wrapped = [FastAPIFileAdapter(f) for f in files]  # Wrap each UploadFile for internal compatibility
        # ChatIngestor: handles chunking documents and saving embeddings into a FAISS vector store
        ci = ChatIngestor(
            temp_base=UPLOAD_BASE,                   # Directory to temporarily store uploaded files
            faiss_base=FAISS_BASE,                   # Root directory for FAISS index storage
            use_session_dirs=use_session_dirs,        # Isolate each session's index in its own subfolder
            session_id=session_id or None,            # Reuse existing session or create a new one
        )
        # Build the FAISS retriever: chunks the documents, generates embeddings, and saves the index
        ci.built_retriver(
            wrapped, chunk_size=chunk_size, chunk_overlap=chunk_overlap, k=k
        )
        log.info(f"Index created successfully for session: {ci.session_id}")
        return {"session_id": ci.session_id, "k": k, "use_session_dirs": use_session_dirs}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Chat index building failed")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")
    

# ---------- CHAT: QUERY ----------
# --- POST /chat/query : Ask a question against a previously built FAISS index (RAG) ---
@app.post("/chat/query")
async def chat_query(
    question: str = Form(...),                     # The user's natural-language question
    session_id: Optional[str] = Form(None),        # Session whose index to query
    use_session_dirs: bool = Form(True),           # Whether indexes are stored in per-session directories
    k: int = Form(5),                              # Number of top-k document chunks to retrieve
) -> Any:
    try:
        log.info(f"Received chat query: '{question}' | session: {session_id}")
        # Validate: session_id is required when using per-session directories
        if use_session_dirs and not session_id:
            raise HTTPException(status_code=400, detail="session_id is required when use_session_dirs=True")

        # Resolve the path to the FAISS index directory for this session
        index_dir = os.path.join(FAISS_BASE, session_id) if use_session_dirs else FAISS_BASE  # type: ignore
        if not os.path.isdir(index_dir):
            raise HTTPException(status_code=404, detail=f"FAISS index not found at: {index_dir}")

        # Load the FAISS index and build a retrieval chain, then invoke the question
        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(index_dir, k=k, index_name=FAISS_INDEX_NAME)  # Load index & build retriever + chain
        response = rag.invoke(question, chat_history=[])  # Run the RAG pipeline; empty history = fresh conversation
        log.info("Chat query handled successfully.")

        return {
            "answer": response,          # The LLM-generated answer grounded in retrieved document chunks
            "session_id": session_id,
            "k": k,
            "engine": "LCEL-RAG"         # Identifies the RAG implementation (LangChain Expression Language)
        }
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Chat query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

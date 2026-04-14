import sys
import os
from operator import itemgetter
from typing import List, Optional, Dict, Any

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.vectorstores import FAISS

from utils.model_loader import Modelloader
from exceptions.custom_exception import DocumentPortalException
from logger import GLOBAL_LOGGER as log
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import PromptType


class ConversationalRAG:

    """LCEL-based Conversational RAG with lazy retriever initialization.
    Usage:
        rag = ConversationalRAG(session_id="abc")
        rag.load_retriever_from_faiss(index_path="faiss_index/abc", k=5, index_name="index")
        answer = rag.invoke("What is ...?", chat_history=[])
    """

    def __init__(self, session_id: Optional[str], retriever=None):
            
        try:
            self.session_id = session_id

            #Load LLM and prompts during initialization
            self.llm = self._load_llm()
            self.contextualize_prompt: ChatPromptTemplate = PROMPT_REGISTRY[PromptType.CONTEXTUALIZE_QUESTION.value]
            self.qa_prompt: ChatPromptTemplate = PROMPT_REGISTRY[PromptType.CONTEXT_QA.value]

            self.retriever = retriever
            self.chain = None
            if self.retriever is not None:
                self._build_lcel_chain()

                log.info("ConversationalRAG initialized with retriever", session=self.session_id)
            else:
                log.error("failed to initialize ConversationalRAG", session=self.session_id)
                raise DocumentPortalException("Failed to initialize ConversationalRAG")

        except Exception as e:
            log.error("Error initializing ConversationalRAG", error=str(e))
            raise DocumentPortalException("Error initializing ConversationalRAG", e) from e

    #----------Public Methods/APIs----------#

    def load_retriever_from_faiss(
        self,
        index_path: str,
        k: int = 5,
        index_name: str = "index",
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Load FAISS vectorstore from disk and build retriever + LCEL chain.
        """
        try:
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index directory not found: {index_path}")
            embeddings = Modelloader().load_embeddings()
            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                index_name=index_name,
            )

            if search_kwargs is None:
                search_kwargs = {"k": k}
            
            self.retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
            self._build_lcel_chain()

            log.info(
                "Retriever loaded from FAISS and LCEL chain built", 
                index_path=index_path, 
                session=self.session_id, 
                index_name=index_name
                )
            return self.retriever


        except Exception as e:
            log.error("Error loading retriever from FAISS", error=str(e), session=self.session_id)
            raise DocumentPortalException("Error loading retriever from FAISS", e) from e

    def invoke(self, user_input: str, chat_history:Optional[List[BaseMessage]] = None) -> str:
        """Invoke the LCEL pipeline."""

        try:
            if self.chain is None:
                raise DocumentPortalException(
                    "RAG chain not initialized. Ensure retriever is loaded and chain is built before invoking."
                )
            chat_history = chat_history or []
            payload = {"input": user_input, "chat_history": chat_history}
            answer = self.chain.invoke(payload)

            if not answer:
                log.warning(
                    "No answer generated", user_input=user_input, session=self.session_id
                )
                return "No answer could be generated"
            log.info(
                "LCEL chain invoked successfully", 
                user_input=user_input, 
                session=self.session_id, 
                answer_preview=str(answer)[:100]
                )
            return answer

        except Exception as e:
            log.error("Error invoking LCEL chain", error=str(e), session=self.session_id)
            raise DocumentPortalException("Error invoking LCEL chain", e) from e

    

#----INTERNALS----


    def _load_llm(self):
        try:
            llm = Modelloader().load_llm()
            if not llm:
                raise ValueError("LLM could not be loaded")
            log.info("LLM loaded successfully", session_id=self.session_id)
            return llm
        except Exception as e:
            log.error("Failed to load LLM", error=str(e))
            raise DocumentPortalException("LLM loading error in ConversationalRAG", e) from e

    @staticmethod
    def _format_docs(docs) -> str:
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    def _build_lcel_chain(self):
        try:
            if self.retriever is None:
                raise DocumentPortalException("No retriever set before building chain", sys)

            # 1) Rewrite user question with chat history context
            question_rewriter = (
                {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )

            # 2) Retrieve docs for rewritten question
            retrieve_docs = question_rewriter | self.retriever | self._format_docs

            # 3) Answer using retrieved context + original input + chat history
            self.chain = (
                {
                    "context": retrieve_docs,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )

            log.info("LCEL graph built successfully", session_id=self.session_id)
        except Exception as e:
            log.error("Failed to build LCEL chain", error=str(e), session_id=self.session_id)
            raise DocumentPortalException("Failed to build LCEL chain", sys)

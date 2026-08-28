import os
import shutil
from pathlib import Path
from typing import Generator, List, Optional
from django.conf import settings
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    WebBaseLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# ── Global Singleton for Embeddings (Loaded once in memory) ──────────

_EMBEDDING_MODEL: Optional[HuggingFaceEmbeddings] = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Returns the shared singleton HuggingFaceEmbeddings model.
    """
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _EMBEDDING_MODEL


# ── User Vector Store Paths ──────────────────────────────────────────

def get_user_vectorstore_dir(user_id: int) -> Path:
    """
    Returns the isolated vector store directory for a specific user.
    Path: media/vectorstores/user_<user_id>/
    """
    path = Path(settings.MEDIA_ROOT) / "vectorstores" / f"user_{user_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_user_vectorstore(user_id: int) -> Optional[FAISS]:
    """
    Loads and returns the user's isolated FAISS vector store,
    or None if no index exists yet.
    """
    store_dir = get_user_vectorstore_dir(user_id)
    index_file = store_dir / "index.faiss"

    if not index_file.exists():
        return None

    try:
        embedding_model = get_embedding_model()
        return FAISS.load_local(
            str(store_dir),
            embedding_model,
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        print(f"Error loading vector store for user {user_id}: {e}")
        return None


# ── Multi-Format Loader Helper ───────────────────────────────────────

def load_documents_by_type(doc_type: str, file_path_or_url: str) -> List[Document]:
    """
    Loads documents using the appropriate LangChain loader based on doc_type.
    Supports: 'pdf', 'text', 'csv', 'web'.
    """
    target = str(file_path_or_url).strip()

    if doc_type == "pdf":
        loader = PyPDFLoader(target)
        return loader.load()

    elif doc_type == "text":
        try:
            loader = TextLoader(target, encoding="utf-8", autodetect_encoding=True)
            return loader.load()
        except Exception:
            loader = TextLoader(target, autodetect_encoding=True)
            return loader.load()

    elif doc_type == "csv":
        try:
            loader = CSVLoader(target, encoding="utf-8")
            return loader.load()
        except Exception:
            loader = CSVLoader(target)
            return loader.load()

    elif doc_type == "web":
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        }
        loader = WebBaseLoader(target, header_template=headers)
        return loader.load()

    else:
        # Fallback to TextLoader
        loader = TextLoader(target, autodetect_encoding=True)
        return loader.load()


# ── Unified Document Ingestion with Progress Generator ───────────────

def process_document_with_progress(
    user_id: int,
    doc_type: str,
    file_path_or_url: str,
    original_title: str,
) -> Generator[dict, None, None]:
    """
    Processes an uploaded file (.pdf, .txt, .csv) or web URL (.web),
    splits into chunks, generates vector embeddings, stores them in the
    user-isolated FAISS database, and yields granular SSE progress events.
    """
    try:
        type_labels = {
            "pdf": "PDF document",
            "text": "Text document",
            "csv": "CSV spreadsheet",
            "web": "Web page",
        }
        type_label = type_labels.get(doc_type, "document")

        # Step 1: Loading Document (15%)
        yield {
            "percent": 15,
            "stage": "reading",
            "message": f"Reading {type_label} '{original_title}'...",
        }

        raw_documents = load_documents_by_type(doc_type, file_path_or_url)
        item_count = len(raw_documents)

        if not raw_documents:
            yield {
                "percent": 100,
                "stage": "error",
                "message": f"The {type_label} contains no readable content.",
            }
            return

        # Step 2: Splitting text into semantic chunks (35%)
        yield {
            "percent": 35,
            "stage": "splitting",
            "message": f"Splitting content into semantic chunks...",
        }

        # For CSV, each row is often already a distinct document; split with overlap
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(raw_documents)

        if not chunks:
            yield {
                "percent": 100,
                "stage": "error",
                "message": "Unable to extract chunks from document content.",
            }
            return

        # Inject user_id and source metadata into chunk metadata
        for chunk in chunks:
            chunk.metadata["user_id"] = str(user_id)
            chunk.metadata["source_name"] = original_title
            chunk.metadata["doc_type"] = doc_type
            if doc_type == "web":
                chunk.metadata["source_url"] = file_path_or_url

        chunk_count = len(chunks)

        # Step 3: Generating Vector Embeddings (65%)
        yield {
            "percent": 65,
            "stage": "embedding",
            "message": f"Generating vector embeddings for {chunk_count} chunks...",
        }

        embedding_model = get_embedding_model()
        store_dir = get_user_vectorstore_dir(user_id)
        existing_store = get_user_vectorstore(user_id)

        # Step 4: Indexing and Saving to User's Vector Store (90%)
        yield {
            "percent": 90,
            "stage": "indexing",
            "message": "Saving vectors into your isolated vector database...",
        }

        if existing_store is not None:
            existing_store.add_documents(chunks)
            existing_store.save_local(str(store_dir))
        else:
            new_store = FAISS.from_documents(chunks, embedding_model)
            new_store.save_local(str(store_dir))

        # Step 5: Completed (100%)
        yield {
            "percent": 100,
            "stage": "done",
            "message": f"Successfully indexed '{original_title}' ({chunk_count} chunks)!",
            "page_count": item_count,
            "chunk_count": chunk_count,
        }

    except Exception as e:
        yield {
            "percent": 100,
            "stage": "error",
            "message": f"Processing failed: {str(e)}",
        }


def process_pdf_with_progress(
    user_id: int,
    file_path: str,
    original_filename: str,
) -> Generator[dict, None, None]:
    """
    Backwards-compatible alias for PDF processing.
    """
    return process_document_with_progress(user_id, "pdf", file_path, original_filename)


# ── Retriever & Context Formatting ───────────────────────────────────

def get_user_retriever(user_id: int, k: int = 5):
    """
    Returns a retriever for the specified user's vector store.
    """
    vector_store = get_user_vectorstore(user_id)
    if vector_store is None:
        return None
    return vector_store.as_retriever(search_kwargs={"k": k})


def format_rag_context(docs: List[Document]) -> str:
    """
    Formats retrieved documents into a context block with source citations.
    Handles PDF pages, CSV rows, web URLs, and text files.
    """
    formatted_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_name", "Document")
        doc_type = doc.metadata.get("doc_type", "")
        
        detail_tag = ""
        if doc_type == "pdf" and "page" in doc.metadata:
            detail_tag = f" (Page {doc.metadata['page'] + 1})"
        elif doc_type == "csv" and "row" in doc.metadata:
            detail_tag = f" (Row {doc.metadata['row'] + 1})"
        elif doc_type == "web" and "source_url" in doc.metadata:
            detail_tag = f" ({doc.metadata['source_url']})"

        header = f"[{source}{detail_tag}]"
        formatted_parts.append(f"{header}\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(formatted_parts)


# ── Document Removal / Re-indexing ───────────────────────────────────

def clear_user_vectorstore(user_id: int):
    """
    Removes the user's vector store directory.
    """
    store_dir = get_user_vectorstore_dir(user_id)
    if store_dir.exists():
        shutil.rmtree(store_dir)


def rebuild_user_vectorstore(user_id: int, remaining_documents):
    """
    Rebuilds the user's vector store from their remaining active documents across all 4 formats.
    """
    clear_user_vectorstore(user_id)
    if not remaining_documents:
        return

    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for doc_record in remaining_documents:
        doc_type = getattr(doc_record, "doc_type", "pdf")
        target_path_or_url = None

        if doc_type == "web" and doc_record.url:
            target_path_or_url = doc_record.url
        elif doc_record.file and os.path.exists(doc_record.file.path):
            target_path_or_url = doc_record.file.path

        if target_path_or_url:
            try:
                raw_docs = load_documents_by_type(doc_type, target_path_or_url)
                chunks = splitter.split_documents(raw_docs)
                for chunk in chunks:
                    chunk.metadata["user_id"] = str(user_id)
                    chunk.metadata["source_name"] = doc_record.title
                    chunk.metadata["doc_type"] = doc_type
                    if doc_type == "web":
                        chunk.metadata["source_url"] = doc_record.url
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error reading {doc_record.title} during rebuild: {e}")

    if all_chunks:
        embedding_model = get_embedding_model()
        store_dir = get_user_vectorstore_dir(user_id)
        new_store = FAISS.from_documents(all_chunks, embedding_model)
        new_store.save_local(str(store_dir))

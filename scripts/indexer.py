from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_PDF_DIR = Path(__file__).resolve().parents[1] / "papers"
DEFAULT_INDEX_DIR = Path(__file__).resolve().parents[1] / "faiss_index"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a local FAISS index from scientific PDFs for RAG."
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=DEFAULT_PDF_DIR,
        help=f"Directory containing PDF files. Default: {DEFAULT_PDF_DIR}",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help=f"Directory where the FAISS index will be saved. Default: {DEFAULT_INDEX_DIR}",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for LangChain text splitting. Default: 1000",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap for LangChain text splitting. Default: 200",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=(
            "Hugging Face sentence-transformer model for embeddings. "
            f"Default: {DEFAULT_EMBEDDING_MODEL}"
        ),
    )
    return parser.parse_args()


def load_pdf_documents(pdf_path: Path) -> list[Document]:
    reader = PdfReader(str(pdf_path))
    documents: list[Document] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(pdf_path),
                    "file_name": pdf_path.name,
                    "page": page_number,
                },
            )
        )

    return documents


def iter_pdf_files(pdf_dir: Path) -> Iterable[Path]:
    return sorted(
        path for path in pdf_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"
    )


def build_vectorstore(
    pdf_dir: Path,
    index_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
) -> None:
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory does not exist: {pdf_dir}")
    if not pdf_dir.is_dir():
        raise NotADirectoryError(f"PDF path is not a directory: {pdf_dir}")
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than 0.")
    if chunk_overlap < 0:
        raise ValueError("Chunk overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    pdf_files = list(iter_pdf_files(pdf_dir))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir}")

    raw_documents: list[Document] = []
    for pdf_file in pdf_files:
        pdf_documents = load_pdf_documents(pdf_file)
        raw_documents.extend(pdf_documents)
        print(
            f"Loaded {len(pdf_documents)} non-empty pages from {pdf_file.name}",
            flush=True,
        )

    if not raw_documents:
        raise ValueError("PDF files were found, but no extractable text was produced.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunked_documents = splitter.split_documents(raw_documents)

    if not chunked_documents:
        raise ValueError("No chunks were created from the extracted PDF text.")

    print(f"Created {len(chunked_documents)} text chunks", flush=True)
    print(f"Loading embedding model: {embedding_model}", flush=True)

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = FAISS.from_documents(chunked_documents, embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))

    print(
        f"Saved FAISS index with {len(chunked_documents)} chunks to {index_dir}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    build_vectorstore(
        pdf_dir=args.pdf_dir.expanduser().resolve(),
        index_dir=args.index_dir.expanduser().resolve(),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
    )


if __name__ == "__main__":
    main()

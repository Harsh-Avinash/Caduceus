import os
import glob
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from multiprocessing import Pool
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    CSVLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from typing import List


class DocumentLoader:
    LOADER_MAPPING = {
        ".csv": (CSVLoader, {}),
        ".doc": (UnstructuredWordDocumentLoader, {}),
        ".docx": (UnstructuredWordDocumentLoader, {}),
        ".pdf": (PyMuPDFLoader, {}),
        ".ppt": (UnstructuredPowerPointLoader, {}),
        ".pptx": (UnstructuredPowerPointLoader, {}),
        ".txt": (TextLoader, {"encoding": "utf8"}),
    }

    @staticmethod
    def load_single_document(file_path: str) -> List[str]:
        ext = "." + file_path.rsplit(".", 1)[-1]
        if ext in DocumentLoader.LOADER_MAPPING:
            loader_class, loader_args = DocumentLoader.LOADER_MAPPING[ext]
            loader = loader_class(file_path, **loader_args)
            documents = loader.load()
            if documents:
                return [doc.page_content for doc in documents]
            else:
                print(f"No documents found in file: {file_path}")
                return []
        else:
            raise ValueError(f"Unsupported file extension '{ext}'")

    @staticmethod
    def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[str]:
        all_files = []
        for ext in DocumentLoader.LOADER_MAPPING:
            all_files.extend(
                glob.glob(os.path.join(
                    source_dir, f"**/*{ext}"), recursive=True)
            )

        filtered_files = [
            file_path for file_path in all_files if file_path not in ignored_files
        ]

        with Pool(processes=os.cpu_count()) as pool:
            results = []
            with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
                for i, docs in enumerate(pool.imap_unordered(DocumentLoader.load_single_document, filtered_files)):
                    results.extend(docs)
                    pbar.update()

        return results


def split_text_into_chunks(texts: List[str], chunk_size: int) -> List[str]:
    chunks = []
    for text in texts:
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i+chunk_size])
    return chunks


def main():
    model = SentenceTransformer(embeddings_model_name)

    documents = DocumentLoader.load_documents(source_directory)
    if not documents:
        print("No new documents to load")
        exit(0)

    print(f"Loaded {len(documents)} new documents from {source_directory}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = split_text_into_chunks(documents, chunk_size)

    print(
        f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")

    embeddings = model.encode(texts)

    # Save embeddings and texts for future use
    np.save(os.path.join(persist_directory, "embeddings.npy"), embeddings)
    with open(os.path.join(persist_directory, "texts.pkl"), "wb") as f:
        pickle.dump(texts, f)

    print("Ingestion complete!")


if __name__ == "__main__":
    load_dotenv()

    persist_directory = os.getenv("PERSIST_DIRECTORY", "db")
    source_directory = os.getenv("SOURCE_DIRECTORY", "input")
    embeddings_model_name = os.getenv(
        "EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
    chunk_size = 500
    chunk_overlap = 50

    main()

from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    CSVLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from tqdm import tqdm
from multiprocessing import Pool
from dotenv import load_dotenv
from typing import List
import glob
import os


def get_embed_model(model_name, model_directory):
    # Create directory if it doesn't exist
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    model_path = os.path.join(model_directory, model_name)

    # Check if model files exist in the directory
    if not os.path.exists(model_path):
        # If model files don't exist, download them and save
        model = SentenceTransformer(model_name)
        model.save(model_path)

    return model_path


print("Script called")
load_dotenv()

# Define the folder for storing database
PERSIST_DIRECTORY = os.environ.get('PERSIST_DIRECTORY')

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl='duckdb+parquet',
    persist_directory=PERSIST_DIRECTORY,  # type: ignore
    anonymized_telemetry=False
)

# Load environment variables
persist_directory = os.environ.get('PERSIST_DIRECTORY')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'input')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
chunk_size = 500
chunk_overlap = 50
MODEL_DIRECTORY = 'models'  # Update this path as per your requirement

LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )

    filtered_files = [
        file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()

    return results


def process_documents(ignored_files: List[str] = []) -> List[Document]:
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)

    print(f"Loaded {len(documents)} new documents from {source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)

    print(
        f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")

    return texts


def main():
    model_path = get_embed_model(embeddings_model_name, MODEL_DIRECTORY)
    embeddings = HuggingFaceEmbeddings(model_name=model_path)

    if os.path.exists(persist_directory):  # type: ignore
        print("Removing existing vectorstore")
        os.system(f"rmdir /S /Q {persist_directory}")
    print("Creating new vectorstore")
    texts = process_documents()
    print(f"Creating embeddings for {len(texts)} chunks of text")
    db = Chroma.from_documents(
        texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None

    print(f"Ingestion complete!")


if __name__ == "__main__":
    main()

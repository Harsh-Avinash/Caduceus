
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
from chromadb.config import Settings
import os
import time
from sentence_transformers import SentenceTransformer

load_dotenv()

embeddings_model_name = os.getenv("EMBEDDINGS_MODEL_NAME")
persist_directory = os.getenv('PERSIST_DIRECTORY')
model_type = os.getenv('MODEL_TYPE')
model_path = os.getenv('MODEL_PATH')
model_n_ctx = os.getenv('MODEL_N_CTX')
model_n_batch = int(os.getenv('MODEL_N_BATCH', 8))
target_source_chunks = int(os.getenv('TARGET_SOURCE_CHUNKS', 4))

CHROMA_SETTINGS = Settings(
    chroma_db_impl='duckdb+parquet',
    persist_directory=persist_directory,  # type: ignore
    anonymized_telemetry=False
)

MODEL_DIRECTORY = 'models'
EMBEDDINGS_MODEL_NAME = 'all-MiniLM-L6-v2'


def get_embed_model(model_name, model_directory):
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    model_path = os.path.join(model_directory, model_name)
    if not os.path.exists(model_path):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        model.save(model_path)
    return model_path


def get_answer(qa, query):
    start = time.time()
    prompt_context = "Answer the above question given that you are an artificial intelligence researcher. If you are not sure about the answer, tell me that you don't know the answer."
    modified_query = query + "\n" + prompt_context
    res = qa(modified_query)
    answer = res['result']
    docs = res['source_documents']
    end = time.time()
    return query, answer, round(end - start, 2), docs


def process_query(query):
    model_path_e = get_embed_model(embeddings_model_name, MODEL_DIRECTORY)
    embeddings = HuggingFaceEmbeddings(model_name=model_path_e)
    db = Chroma(persist_directory=persist_directory,
                embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    callbacks = [StreamingStdOutCallbackHandler()]
    local_path = "models/ggml-gpt4all-j-v1.3-groovy.bin"
    llm = GPT4All(model=local_path, callbacks=callbacks,
                  verbose=True)  # type: ignore
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    res = get_answer(qa, query)
    return res


if __name__ == "__main__":
    while True:
        query_input = input("\nEnter a query: ")
        if query_input == "exit":
            break
        if query_input.strip() == "":
            continue
        result = process_query(query_input)
        print("\n\n> Question:")
        print(result[0])
        print(f"\n> Answer (took {result[2]} s.):")
        print(result[1])
        for doc in result[3]:
            print("\n> " + doc.metadata["source"] + ":")
            print(doc.page_content)

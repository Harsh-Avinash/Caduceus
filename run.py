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
    persist_directory=persist_directory,
    anonymized_telemetry=False
)


def get_embed_model(model_name, model_directory):
    # Create directory if it doesn't exist
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    model_path = os.path.join(model_directory, model_name)

    # Check if model files exist in the directory
    if not os.path.exists(model_path):
        # If model files don't exist, download them and save
        model = SentenceTransformer('all-MiniLM-L6-v2')
        model.save(model_path)

    return model_path


MODEL_DIRECTORY = 'models'
EMBEDDINGS_MODEL_NAME = 'all-MiniLM-L6-v2'


# For GUI
def get_answer(qa, query):
    start = time.time()
    # Define a prompt context that introduces the context of the query
    prompt_context = "Answer the above question given that you are an artificial intelligence researcher. If you are not sure about the answer, tell me that you don't know the answer."
    modified_query = query + "\n" + prompt_context
    res = qa(modified_query)
    answer = res['result']
    docs = res['source_documents']
    end = time.time()

    return query, answer, round(end - start, 2), docs


def main():
    model_path_e = get_embed_model(embeddings_model_name, MODEL_DIRECTORY)
    embeddings = HuggingFaceEmbeddings(model_name=model_path_e)

    db = Chroma(persist_directory=persist_directory,
                embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    callbacks = [StreamingStdOutCallbackHandler()]
    llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj',
                  n_batch=model_n_batch, callbacks=callbacks, verbose=False)

    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    # Uncomment the return statement below to use in GUI mode
    return qa

    while True:
        query = input("\nEnter a query: ")
        print("Question received: " + query + "\nPlease wait...")
        if query == "exit":
            print("Exiting...")
            break
        if query.strip() == "":
            continue

        start = time.time()
        res = get_answer(qa, query)
        answer = res[1]
        end = time.time()

        print("\n\n> Question:")
        print(query)
        print(f"\n> Answer (took {round(end - start, 2)} s.):")
        print(answer)

        # print the source documents if they are returned
        sourcedocs = res[3]

        for sourcedoc in sourcedocs:
            print("\n> " + sourcedoc.metadata["source"] + ":")
            print(sourcedoc.page_content)


if __name__ == "__main__":
    main()

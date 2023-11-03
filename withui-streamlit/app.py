import streamlit as st
from qa import get_answer
import os
from langchain.vectorstores import Chroma

# Define the directory for storing vectorstore
VECTORSTORE_DIRECTORY = 'vectorstore_directory'

# Check if the vectorstore directory exists
setup_exists = os.path.exists(VECTORSTORE_DIRECTORY)

# Function to create and persist vectorstore


def create_vectorstore():
    from langchain.prompts import PromptTemplate
    from langchain.llms import GPT4All
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    from langchain.document_loaders import DirectoryLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import Chroma

    local_path = "models/ggml-gpt4all-j-v1.3-groovy.bin"
    callbacks = [StreamingStdOutCallbackHandler()]
    llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)

    loader = DirectoryLoader('input', glob="**/*.pdf")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(documents=all_splits)

    # Assuming there's a method to save the vectorstore to a directory
    vectorstore.persist(VECTORSTORE_DIRECTORY)  # Adjusted this line

    return {
        "llm": llm,
        "vectorstore": vectorstore
    }


# Load the vectorstore if directory exists
if setup_exists:
    vectorstore = Chroma.from_directory(VECTORSTORE_DIRECTORY)
    setup_objects = {
        "llm": None,  # You'll need to handle how you want to store and retrieve the LLM model
        "vectorstore": vectorstore
    }
else:
    setup_objects = None

chat_log = []

st.title('Question Answering System')

if setup_exists:
    if st.button('Refresh Database (Create New VectorStore)'):
        setup_objects = create_vectorstore()
else:
    if st.button('Setup Database (Create VectorStore)'):
        setup_objects = create_vectorstore()

# Display previous chat history
for q, a in chat_log:
    st.write(f"Q: {q}")
    st.write(f"A: {a}")

# Text input for question
question = st.text_input('Enter your question:')

if question and setup_objects:
    with st.spinner('Fetching answer...'):
        answer = get_answer(
            question, setup_objects["llm"], setup_objects["vectorstore"])
        chat_log.append((question, answer))
        st.write(f"Q: {question}")
        st.write(f"A: {answer}")

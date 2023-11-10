from langchain.chains import RetrievalQA


def get_answer(question, llm, vectorstore):
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever()
    )
    return qa_chain({"query": question})

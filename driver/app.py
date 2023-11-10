
import streamlit as st
from run_refactored import process_query
import subprocess

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title("Document Vector Embedding and Query Interface")

# Create a sidebar for navigation
page = st.sidebar.radio(
    "Choose a page", ["Home", "Refresh Vector Store", "Question Answering"])

if page == "Home":
    st.header("Welcome!")
    st.write(
        "This interface allows you to:"
        "- Refresh the vector store for your documents."
        "- Query the system to get answers based on the embedded documents."

        "Navigate to the desired page using the sidebar."
    )

elif page == "Refresh Vector Store":
    st.header("Refresh Vector Store")
    st.write(
        "Use this page to run the `ingest1.py` script which refreshes the vector store for your documents."
    )
    if st.button("Run ingest1.py"):
        result = subprocess.run(["python", "ingest1.py"], capture_output=True)
        st.text(result.stdout.decode())

elif page == "Question Answering":
    st.header("Ask a Question")
    st.write("Enter your query below and press 'Submit' to get an answer.")

    user_query = st.text_input("Enter your query:")
    if st.button("Submit Query"):
        res = process_query(user_query)
        answer = res[1]

        # Update chat history
        st.session_state.chat_history.append(("You", user_query))
        st.session_state.chat_history.append(("AI", answer))

        # Display chat history
        for role, msg in st.session_state.chat_history:
            st.write(f"**{role}**: {msg}")

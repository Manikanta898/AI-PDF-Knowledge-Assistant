
"""
app.py

Purpose:
    Streamlit frontend for the AI PDF Knowledge Assistant.

Current functionality:
    1. Upload a PDF.
    2. Extract text and page metadata.
    3. Split the extracted content into chunks.
    4. Generate embeddings and create a FAISS vector store.
    5. Save the vector store locally.
"""

import os
import streamlit as st

# Import the backend functions we have already developed.
from utils.pdf_loader import load_pdf_pages
from utils.text_splitter import split_documents
from utils.vector_store import create_vector_store, save_vector_store
from utils.vector_store import search_similar_chunks
from utils.chatbot import generate_answer

# --------------------------------------------------
# 1. Configure the Streamlit application
# --------------------------------------------------

st.set_page_config(
    page_title="AI PDF Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)

# Display the application title and description.
st.title("📚 AI PDF Knowledge Assistant")

st.write(
    "Upload a PDF document and ask questions "
    "based on its content."
)


# --------------------------------------------------
# 2. Create the PDF upload component
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"]
)


# --------------------------------------------------
# 3. Process the uploaded PDF
# --------------------------------------------------

if uploaded_file is not None:

    # Display the uploaded filename.
    st.info(f"Uploaded file: {uploaded_file.name}")

    # Process the PDF only when the user clicks this button.
    if st.button("Process PDF"):

        with st.spinner("Processing your PDF..."):

            # Create the folder if it doesn't already exist.
            os.makedirs("uploaded_files", exist_ok=True)

            # Use only the filename, not any directory supplied with it.
            pdf_name = os.path.basename(uploaded_file.name)

            # Define where the uploaded PDF will be stored.
            pdf_path = os.path.join("uploaded_files", pdf_name)

            # Save the uploaded PDF to the local folder.
            with open(pdf_path, "wb") as file:
                file.write(uploaded_file.getbuffer())

            # Step 1: Extract page content and metadata.
            documents = load_pdf_pages(pdf_path)

            if not documents:
                st.error(
                    "No readable text found in the PDF. "
                    "It may be scanned or image-based."
                )
                st.stop()

            # Step 2: Split the documents into smaller chunks.
            chunks = split_documents(documents)

            if not chunks:
                st.error("No text chunks were generated.")
                st.stop()

            # Step 3: Generate embeddings and create FAISS index.
            vector_store = create_vector_store(chunks)

            # Step 4: Save the vector store locally.
            save_vector_store(vector_store, "vector_db")

            # Keep the vector store available during this session.
            st.session_state.vector_store = vector_store
            st.session_state.pdf_processed = True

        # Display processing results.
        st.success("PDF processed successfully!")

        st.write(f"**PDF pages extracted:** {len(documents)}")
        st.write(f"**Text chunks created:** {len(chunks)}")

# --------------------------------------------------
# 4. Chat interface
# --------------------------------------------------

# Display the chat section only after a PDF is processed.
if st.session_state.get("pdf_processed", False):

    st.divider()
    st.header("💬 Chat with your PDF")

    # Initialize chat history for the current session.
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display previous messages.
    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create a chat input box.
    user_question = st.chat_input("Ask a question about your PDF...")

    # Process the user's question.
    if user_question:

        # Display and save the user's question.
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate an answer using the RAG pipeline.
        with st.chat_message("assistant"):

            with st.spinner("Searching your PDF and generating an answer..."):

                # Step 1: Retrieve relevant chunks from FAISS.
                results = search_similar_chunks(
                    vector_store=st.session_state.vector_store,
                    query=user_question,
                    k=3
                )

                # Step 2: Extract Document objects from the results.
                retrieved_docs = [
                    document for document, score in results
                ]

                # Step 3: Generate the answer using Gemini.
                response = generate_answer(
                    query=user_question,
                    retrieved_docs=retrieved_docs
                )

                answer = response["answer"]

                # Step 4: Display the generated answer.
                st.markdown(answer)

                # Step 5: Display the retrieved source references.
                with st.expander("View source references"):

                    for source in response["sources"]:
                        st.write(
                            f'{source["source_id"]}: '
                            f'{source["source"]}, '
                            f'Page {source["page"]}'
                        )

        # Save the assistant's answer in chat history.
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })

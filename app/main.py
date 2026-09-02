"""
main.py
The Streamlit app — this is what the user actually interacts with.
Upload a PDF -> ask questions -> get answers with sources.
"""

import streamlit as st
import tempfile
import os
from ingest import process_document
from vectorstore import VectorStore
from generate import generate_answer

st.set_page_config(page_title="DocMind — Chat with your PDF", page_icon="📄")
st.title("📄 DocMind — Chat with your Documents")
st.caption("Upload a PDF, ask questions, get grounded answers with sources.")

# Keep the vector store alive across reruns using session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False 

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None and not st.session_state.doc_loaded:
    with st.spinner("Processing document..."):
        # save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        chunks = process_document(tmp_path)
        st.session_state.vector_store.reset()
        st.session_state.vector_store.add_chunks(chunks)
        st.session_state.doc_loaded = True
        os.remove(tmp_path)

    st.success(f"Document processed into {len(chunks)} chunks. Ask away!")

if st.session_state.doc_loaded:
    question = st.text_input("Ask a question about the document:")

    if question:
        with st.spinner("Searching document and generating answer..."):
            relevant_chunks = st.session_state.vector_store.search(question, top_k=3)
            answer = generate_answer(question, relevant_chunks)

        st.markdown("### Answer")
        st.write(answer)

        with st.expander("View source chunks used"):
            for c in relevant_chunks:
                st.markdown(f"**{c['id']}**")
                st.text(c["text"][:400] + "...")

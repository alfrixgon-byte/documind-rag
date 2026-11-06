import streamlit as st
import tempfile
import os
from src.rag_chain import AdvancedRAGPipeline

st.set_page_config(page_title="DocuMind RAG System", layout="wide")
st.title("📄 DocuMind: Enterprise RAG System")

api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
uploaded_file = st.sidebar.file_uploader("Upload a PDF Document", type=["pdf"])

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None

if uploaded_file and api_key:
    if st.sidebar.button("Process Document"):
        with st.spinner("Indexing document & building vector database..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            pipeline = AdvancedRAGPipeline(openai_api_key=api_key)
            pipeline.ingest_document(tmp_path)
            st.session_state.rag_pipeline = pipeline
            os.remove(tmp_path)
            st.sidebar.success("Document successfully indexed!")

query = st.text_input("Ask a question about the document:")
if query and st.session_state.rag_pipeline:
    with st.spinner("Retrieving context & generating answer..."):
        result = st.session_state.rag_pipeline.query(query)
        st.markdown("### Answer:")
        st.write(result["answer"])

        st.markdown("### Source References:")
        for doc in result["source_documents"]:
            with st.expander(f"Page {doc.metadata.get('page', 'N/A')}"):
                st.write(doc.page_content)


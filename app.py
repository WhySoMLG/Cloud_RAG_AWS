import streamlit as st
import tempfile
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from rag_connector import RAGConnector

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Multimodal RAG", page_icon="🧠", layout="wide")
st.title("🧠 Cloud-Optimized Multimodal RAG")
st.markdown("Upload documents or images. Powered by GitHub Models (GPT-4o-mini) and Pinecone.")

# --- 2. SESSION STATE SETUP ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag" not in st.session_state:
    with st.spinner("Connecting to Pinecone Vector DB and GitHub Models..."):
        try:
            st.session_state.rag = RAGConnector()
        except ValueError as e:
            st.error(f"Configuration Error: {e}")
            st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. SIDEBAR: DATA INGESTION ---
with st.sidebar:
    st.header("📥 Ingest Data")
    st.info("Supported: PDFs, DOCX, TXT, Images.")
    
    uploaded_file = st.file_uploader("Upload a file:")
    
    if uploaded_file is not None:
        if st.button("Process & Index File"):
            with st.spinner(f"Processing `{uploaded_file.name}`..."):
                file_extension = Path(uploaded_file.name).suffix
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                    tmp_path = tmp_file.name
                
                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = uploaded_file.read(8192) 
                        if not chunk:
                            break
                        f.write(chunk)
                
                try:
                    st.session_state.rag.index(
                        file_path=tmp_path, 
                        session_id=st.session_state.session_id, 
                        mime_type=uploaded_file.type
                    )
                    st.success(f"Indexed {uploaded_file.name} successfully!")
                except Exception as e:
                    st.error(f"Error during ingestion: {e}")
                finally:
                    os.unlink(tmp_path)

# --- 4. CHAT INTERFACE ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a question about your files:")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Retrieving from Pinecone and reasoning with GPT-4o-mini..."):
            try:
                results = st.session_state.rag.query(
                    question=prompt, 
                    session_id=st.session_state.session_id,
                    top_k=5
                )
                answer = results["answer"]
                sources = results["sources"]
                
                st.markdown(answer)
                
                if sources:
                    with st.expander("📚 View Sources"):
                        for i, src in enumerate(sources, 1):
                            meta = src.get("metadata", {})
                            source_name = meta.get('source', 'Unknown')
                            st.caption(f"**[{i}] {source_name}** — Similarity Score: `{src['score']:.3f}`")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Error: {e}")
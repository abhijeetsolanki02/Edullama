import streamlit as st
import tempfile
import os 
from dotenv import load_dotenv

# from langchain_community.document_loaders import PyPDFLoaders
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq

load_dotenv()

# Load Groq API Key
groq_api = os.getenv("GROQ_API_KEY")

# Initialize LLM(Groq)
llm = ChatGroq(groq_api_key = groq_api, model_name = "gemma2-9b-it")

# Page title
st.set_page_config(page_title = "Edullama")

# Sidebar
with st.sidebar:
    st.title("EduLlama🦙📖")
    st.markdown("Upload Reference PDF file and PYQs")
    
    uploaded_file_ref = st.file_uploader("Choose a Reference file", type = ["pdf", "jpg", "jpeg", "png"], accept_multiple_files= True)
    uploaded_file_pyq = st.file_uploader("Choose PYQ file", type = ["pdf", "jpg", "jpeg", "png"], accept_multiple_files= True)
    
    if uploaded_file_ref is not None:
        with tempfile.NamedtemporaryFile(delete = False, suffix = ".pdf") as tmp_file:
            tmp_file.write(uploaded_file_ref.get_value())
            tmp_file_path = tmp_file.name
            
        st.success(f"File '{uploaded_file_ref.name}' uploaded successfully")
    if uploaded_file_pyq is not None:
        with tempfile.NamedtemporaryFile(delete = False) as tmp_file_pyq:
            tmp_file_pyq.write(uploaded_file_pyq.get_value())
            tmp_file_pyq_path = tmp_file_pyq.name
            
        st.success(f"File '{uploaded_file_pyq.name}' uploaded successfully")

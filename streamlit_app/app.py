import os 
import streamlit as st 
from dotenv import load_dotenv
import tempfile

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS 
from langchain_groq import ChatGroq

# Groq API
groq_api_key = os.getenv("GROQ_API_KEY")

# Groq Initiation
llm = ChatGroq(groq_api_key = groq_api_key, 
                model_name = "gemma2-9b-it")
 
prompt_template = ChatPromptTemplate.from_template("""
                                          Answer the questions based on the provided context only.
                                          Please provide the most accurate response based on the question.
                                          <context>
                                          {context}
                                          <context>
                                          Question: {input}""")

def vector_embedding(file_path):
    if "vectors" not in st.session_state:
        st.session_state.loader = PyPDFLoader(file_path)
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.embeddings = OllamaEmbeddings(model = "mxbai-embed-large", base_url = "http://localhost:11434")
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        
        
# app title
st.set_page_config(page_title="EduLlama")

# Sidebar
with st.sidebar:
    st.title("EduLlama🦙📖")
    
    uploaded_file_ref = st.file_uploader("Upload Reference file(s)", type= ["pdf", "jpeg", "jpg", "png"])
    uploaded_file_pyq = st.file_uploader("Upload PYQ file(s)", type= ["pdf", "jpeg", "jpg", "png"])
    
    
    if uploaded_file_ref is not None:
        uploaded_file_ref_name = uploaded_file_ref.name
            
        with st.spinner("Processing PDF... This might take a moment."):
            with tempfile.NamedTemporaryFile(delete = False, suffix = ".pdf") as temp_file:
                temp_file.write(uploaded_file_ref.read())
                temp_file_path = temp_file.name

            vector_embedding(temp_file_path)
            st.success("Document Embedded Successfully! You may start asking questions.")
        
st.title("EduLlama🦙💬")        
st.caption("🚀 A Streamlit chatbot powered by AI")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
        
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
         
if prompt := st.chat_input():
    if not uploaded_file_ref:
        st.info("Please add your reference file in the sidebar")
        st.stop()
    
    st.info("Document Embedded Sucessfully!! You may start asking questions")

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": prompt})
    answer = response["answer"]
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)
  
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("----------------------------------------")

        
# input = st.text_input("Ask any Question!")

# if st.button("Document Embedding"):
#     vector_embedding(temp_file_path)
#     st.write("You may start asking questions")
    
# if input:
#     document_chain = create_stuff_documents_chain(llm, prompt)
#     retriever = st.session_state.vectors.as_retriever()
#     retrieval_chain = create_retrieval_chain(retriever, document_chain)
#     response = retrieval_chain.invoke({"input": input})
#     st.write(response["answer"])
    
#     # With a Streamlit expander
#     with st.expander("Document Similarity Search"):
#         for i, doc in enumerate(response["context"]):
#             st.write(doc.page_content)
#             st.write("----------------------------------------")




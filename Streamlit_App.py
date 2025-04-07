import os
from dotenv import load_dotenv, find_dotenv

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# --- Load API Key from environment or Streamlit secrets ---
load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API key not found. Please add it to your .env or Streamlit secrets.")
    st.stop()

# --- Streamlit App ---
st.set_page_config(page_title="RAG Q&A PDF App")
st.title("📄 RAG-Based Q&A App")
st.markdown("Upload a PDF and ask questions based on its content.")

# --- PDF Upload ---
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
qa_chain = None

if uploaded_file:
    with st.spinner("Processing PDF..."):
        # Save the uploaded file to a temp location
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load and split the document
        loader = PyPDFLoader("temp.pdf")
        raw_documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(raw_documents)

        # Vectorize and create retriever
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = FAISS.from_documents(documents, embeddings)
        retriever = vectorstore.as_retriever()

        # Setup LLM and chain
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        st.success("PDF processed. Ask a question below.")

# --- Q&A Interface ---
if qa_chain:
    question = st.text_input("Ask a question about the document:")
    if question:
        with st.spinner("Generating answer..."):
            result = qa_chain({"query": question})
            st.markdown("### ✅ Answer")
            st.write(result["result"])

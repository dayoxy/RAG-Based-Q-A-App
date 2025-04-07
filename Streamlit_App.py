# App.py

import os
from dotenv import load_dotenv, find_dotenv

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI


import gradio as gr
from langchain.chains import RetrievalQA

# --- Load API Key ---
env_path = find_dotenv()
if not env_path:
    print("❌ .env file not found.")
else:
    print("✅ .env loaded from:", env_path)

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("❌ OpenAI API Key not found.")
else:
    print("✅ API Key loaded.")

# --- Global QA Chain ---
qa_chain = None

# --- Load and Chunk PDF ---
def load_pdf(file):
    loader = UnstructuredPDFLoader(file.name)
    return loader.load_and_split()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(pages)

# --- Set Up Retrieval QA ---
def setup_qa(docs):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# --- Upload Handler ---
def upload_pdf(pdf):
    global qa_chain
    docs = load_pdf(pdf)
    qa_chain = setup_qa(docs)
    return "✅ PDF uploaded and ready! Ask your questions below."

# --- Ask Question Handler ---
def answer_question(question):
    if qa_chain:
        result = qa_chain({"query": question})
        return result["result"]
    return "❌ Please upload a PDF first."

# --- Gradio Interface ---
def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# 📄 RAG-Based Q&A App\nUpload a PDF and ask questions.")

        file_input = gr.File(label="Upload PDF")
        upload_btn = gr.Button("Process Document")
        status_box = gr.Textbox(label="Status")

        question_input = gr.Textbox(label="Ask a question")
        answer_output = gr.Textbox(label="Answer")
        ask_btn = gr.Button("Get Answer")

        upload_btn.click(fn=upload_pdf, inputs=file_input, outputs=status_box)
        ask_btn.click(fn=answer_question, inputs=question_input, outputs=answer_output)

    return demo

# --- Launch App ---
if __name__ == "__main__":
    app = create_interface()
    app.launch()

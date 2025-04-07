# 📄 RAG-Based Q&A App

This is a Retrieval-Augmented Generation (RAG) app that allows users to upload a PDF and ask questions about its content using OpenAI's GPT-3.5. Built with LangChain, FAISS, and Streamlit.

## 🚀 Features

- Upload any PDF document
- Ask natural language questions about its content
- Retrieves relevant document chunks using FAISS
- Generates answers using OpenAI's GPT-3.5
- Deployable on Streamlit Cloud

## 🧠 Tech Stack

- Python
- LangChain
- OpenAI API
- FAISS (vector store)
- Streamlit (UI)
- PDF parsing (Unstructured loader)

## 📦 Setup

1. Clone the repo:
```bash
git clone https://github.com/your-username/rag-qa-app.git
cd rag-qa-app
```

2. Create a `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run locally:
```bash
streamlit run streamlit_app.py
```

## ☁️ Streamlit Deployment

Create a file at `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

Then deploy it to [Streamlit Cloud](https://streamlit.io/cloud) using this repo.

## 📄 Example PDFs

You can test with files in the `sample_docs/` folder or your own PDF files.

## 📜 License

MIT

---

Made with ❤️ by [Adedayo Oguntonade](https://github.com/dayoxy)
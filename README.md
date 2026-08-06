# RAG Studio

An end-to-end Retrieval-Augmented Generation (RAG) application that enables users to upload documents, process them into semantic embeddings, and ask natural language questions using Large Language Models (LLMs).

**Live Demo:** https://ragstudio-revpyuepanmwkh2yxg3dfh.streamlit.app/

---

## Features

-  Upload multiple document formats
  - PDF
  - DOCX
  - TXT
  - CSV
  - XLSX
  - HTML
  - Markdown

-  Multiple Chunking Strategies
  - Recursive Chunking
  - Character Chunking
  - Semantic Chunking

-  Configurable LLMs
  - Llama 3.1 8B Instant
  - Llama 3.3 70B
  - Mixtral 8x7B
  - Gemma 2

-  Semantic Search using FAISS

-  Context-aware Question Answering

-  Interactive Chat Interface

-  Adjustable Parameters
  - Chunk Size
  - Chunk Overlap
  - Top-K Retrieval
  - Temperature

-  Resource Usage Dashboard

---

## Tech Stack

### Frontend
- Streamlit

### Backend
- Python

### AI & Machine Learning
- LangChain
- HuggingFace Embeddings
- Groq API
- FAISS
- Sentence Transformers

### Libraries
- Pandas
- NumPy
- PyPDF
- python-docx
- BeautifulSoup4

---

## Project Structure

```
Rag_Studio
│
├── app.py
├── backend.py
├── requirements.txt
├── README.md
└── data/
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/dpkaru/Rag_Studio.git
```

Move into the project

```bash
cd Rag_Studio
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```
GROQ_API_KEY=your_api_key
```

Run the application

```bash
streamlit run app.py
```

---

## Application Preview

<img width="100%" alt="RAG Studio" src="https://raw.githubusercontent.com/dpkaru/Rag_Studio/main/assets/home.png">
<img width="1917" height="870" alt="image" src="https://github.com/user-attachments/assets/608babde-1484-400e-98b5-910ab48ee92d" />

---

## Learning Outcomes

This project helped me gain hands-on experience with:

- Retrieval-Augmented Generation (RAG)
- Prompt Engineering
- Vector Databases
- Document Processing
- Semantic Search
- Large Language Models
- Streamlit Application Development
- LangChain Framework

---

## Author

**Dhruvi Prakash Karu**

- GitHub: https://github.com/dpkaru

---

## 📄 License

This project is developed for educational and portfolio purposes.

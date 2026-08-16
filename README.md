# 📚 PageMind AI — Comprehensive RAG System & Document Assistant

**PageMind AI** is a Retrieval-Augmented Generation (RAG) system that allows users to upload documents and ask natural-language questions about their content.

The system retrieves relevant information from the uploaded document and uses a Large Language Model (LLM) to generate answers grounded in the retrieved context.

This project demonstrates several important concepts in the **LangChain + RAG ecosystem**, including:

* PDF and text document ingestion
* Text chunking and preprocessing
* Vector embeddings
* ChromaDB vector storage
* Similarity Search
* Maximum Marginal Relevance (MMR)
* Multi-Query Retrieval
* Web document loading
* Streamlit-based interactive UI
* Mistral and other LLM integrations

---

## 🎯 Project Overview

Traditional LLM applications rely only on the knowledge contained in the model's training data.

A RAG system follows a different approach:

```text
User Question
      ↓
Document Retrieval
      ↓
Relevant Chunks
      ↓
LLM + Retrieved Context
      ↓
Grounded Answer
```

With **PageMind AI**, users can upload a book or PDF and ask questions such as:

> "What is gradient descent?"

The system searches the document for relevant sections and generates an answer based on the retrieved content.

---

# 🏗️ Project Structure

```text
PageMind-AI/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── app.py                         # Streamlit web application
├── main.py                        # Application/testing entry point
├── create_database.py             # Database generation/testing script
│
├── document loaders/              # Document ingestion examples
│   ├── notes.txt                  # Sample text document
│   ├── pages.py                   # Web page loader
│   ├── pdf.py                     # PDF loader and text splitter
│   └── test.py                    # Text loading and splitting tests
│
├── retrievers/                    # Advanced retrieval strategies
│   ├── arixv.py                   # arXiv retrieval/API integration
│   ├── mmr.py                     # MMR vs similarity search
│   └── multiquery.py              # Multi-query retrieval
│
└── Vector Store/                  # Vector database examples
    └── DB.py                      # ChromaDB initialization/querying
```

> **Note:** `.venv/`, `.env`, `chroma_db/`, and uploaded PDF files are intentionally excluded from Git using `.gitignore`.

---

# 🧩 Core Modules

## 1. Main Web Application — `app.py`

`app.py` powers the interactive Streamlit application.

The application allows the user to:

1. Upload a PDF
2. Extract the document text
3. Split the document into chunks
4. Generate embeddings
5. Store the embeddings in a vector database
6. Retrieve relevant chunks
7. Ask questions about the document
8. Generate an AI-powered answer

---

## 🔹 Embedding Model

The application uses Hugging Face embeddings:

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

The embedding model converts text into dense numerical vectors.

These vectors allow the vector database to compare semantic meaning rather than relying only on exact keyword matches.

For example:

```text
"How does gradient descent work?"
```

can retrieve a document section containing:

```text
"Gradient descent is an optimization algorithm used to minimize a loss function..."
```

even when the wording is different.

---

# 📄 Document Processing

After a PDF is uploaded, the system loads it using `PyPDFLoader`.

```python
loader = PyPDFLoader(file_path)
docs = loader.load()
```

The extracted document is then split into smaller chunks:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)
```

### Why chunking?

Large documents cannot always be passed directly to an LLM.

Chunking divides the document into smaller pieces while keeping enough context for retrieval.

### `chunk_size=1000`

Each chunk contains approximately 1000 characters.

### `chunk_overlap=200`

Each new chunk shares approximately 200 characters with the previous chunk.

This reduces the possibility of losing context when important information falls near a chunk boundary.

---

# 🔎 Retrieval System

PageMind AI uses retrieval techniques to find the most relevant parts of a document.

Example:

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)
```

### Parameters

| Parameter         | Meaning                                  |
| ----------------- | ---------------------------------------- |
| `k=4`             | Number of final documents returned       |
| `fetch_k=10`      | Number of candidate documents considered |
| `lambda_mult=0.5` | Balance between relevance and diversity  |

---

# 🤖 LLM Integration

The retrieved context is passed to an LLM such as Mistral:

```python
from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI(
    model="mistral-small-2506"
)
```

The LLM uses the retrieved document context to generate the final answer.

This helps reduce the chance of answering from unrelated information.

---

# 🧠 Advanced Retrievers

## 2. Multi-Query Retrieval — `retrievers/multiquery.py`

A user's question may not use the same wording as the document.

For example:

```text
"What is gradient descent?"
```

The document might instead use:

```text
"How does iterative optimization minimize a loss function?"
```

Multi-Query Retrieval solves this problem by using an LLM to generate multiple versions of the user's question.

Example:

```python
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)

docs = multi_query_retriever.invoke(
    "What is gradient descent?"
)
```

### How it works

```text
Original Question
       ↓
LLM generates alternative questions
       ↓
Multiple retrieval searches
       ↓
Combined results
       ↓
More complete context
```

This can improve recall when the wording of the question does not match the wording inside the document.

---

# 🔄 MMR vs Similarity Search — `retrievers/mmr.py`

The project also demonstrates the difference between standard similarity search and Maximum Marginal Relevance.

### Similarity Search

```python
similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
```

Similarity search returns the chunks that are closest to the query embedding.

However, several retrieved chunks may contain almost identical information.

### MMR Search

```python
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3}
)
```

MMR attempts to balance:

* Relevance to the query
* Diversity among retrieved chunks

Conceptually:

```text
Query
 ↓
Find relevant documents
 ↓
Choose highly relevant document
 ↓
Penalize redundant documents
 ↓
Choose diverse relevant documents
```

This can provide the LLM with broader context.

---

# 🌐 Document Ingestion

## 3. Web Page Loader — `document loaders/pages.py`

The project demonstrates loading content from web pages using LangChain's web loader.

Example:

```python
from langchain_community.document_loaders import WebBaseLoader

docs = data.load()
```

The loader retrieves content from a URL and converts it into LangChain `Document` objects.

These documents can then be:

```text
Loaded
 ↓
Split
 ↓
Embedded
 ↓
Stored
 ↓
Retrieved
```

---

# 📑 PDF Loader — `document loaders/pdf.py`

The PDF loader demonstrates how PDF documents can be loaded and split into smaller chunks.

Example:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len
)
```

This example intentionally uses smaller chunks to demonstrate how chunk size affects document processing.

`length_function=len` means the splitter measures chunk size based on characters.

---

# 🗄️ Vector Store — `Vector Store/DB.py`

The project also demonstrates creating a vector database using ChromaDB.

Example:

```python
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma_db"
)
```

The vector store contains:

* Document chunks
* Embeddings
* Metadata

Example metadata:

```python
{
    "source": "AI_book"
}
```

The vector database allows semantic search over the stored document chunks.

---

# 🔁 Complete RAG Pipeline

The complete PageMind AI pipeline can be summarized as:

```text
             ┌─────────────────┐
             │   PDF / Document│
             └────────┬────────┘
                      ↓
             ┌─────────────────┐
             │ Document Loader │
             └────────┬────────┘
                      ↓
             ┌─────────────────┐
             │ Text Chunking   │
             └────────┬────────┘
                      ↓
             ┌─────────────────┐
             │    Embeddings   │
             └────────┬────────┘
                      ↓
             ┌─────────────────┐
             │    ChromaDB     │
             └────────┬────────┘
                      ↓
              User asks question
                      ↓
             ┌─────────────────┐
             │    Retriever    │
             └────────┬────────┘
                      ↓
             Relevant Documents
                      ↓
             ┌─────────────────┐
             │       LLM       │
             └────────┬────────┘
                      ↓
                 Final Answer
```

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/ImNirajsingh/PageMind-AI.git
cd PageMind-AI
```

---

## 2. Create a Virtual Environment

Creating a virtual environment isolates project dependencies.

### Windows

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

---

# 📦 Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create a `.env` file in the project root.

Example:

```env
MISTRAL_API_KEY=your_mistral_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
GOOGLE_API_KEY=your_google_api_key
```

Depending on the modules you use, additional API credentials may be required.

> ⚠️ **Never commit your `.env` file or expose API keys publicly.**

The `.env` file is intentionally excluded through `.gitignore`.

---

# ▶️ Run the Application

Make sure your virtual environment is activated.

Then run:

```bash
streamlit run app.py
```

Streamlit will start the application and provide a local URL such as:

```text
http://localhost:8501
```

Open the URL in your browser.

---

# 💻 Using PageMind AI

### Step 1 — Upload a PDF

Select a book, research paper, textbook, or other supported PDF.

### Step 2 — Process the Document

The application:

```text
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Embedding Generation
 ↓
Vector Store
```

### Step 3 — Ask a Question

Example:

```text
What is backpropagation?
```

### Step 4 — Retrieve Relevant Context

The retriever searches the vector database for relevant document chunks.

### Step 5 — Generate the Answer

The LLM uses the retrieved context to produce the response.

---

# ⚠️ Troubleshooting

## `ModuleNotFoundError: No module named 'torchvision'`

This can occur when the Hugging Face embedding stack requires PyTorch-related dependencies.

Try:

```bash
pip install torch torchvision torchaudio
```

For GPU-specific installations, use the appropriate installation command for your CUDA configuration from the official PyTorch installation instructions.

---

## ChromaDB / SQLite Compatibility

Some ChromaDB environments may encounter SQLite compatibility issues.

One possible workaround is:

```bash
pip install pysqlite3-binary
```

Then, before importing ChromaDB:

```python
import pysqlite3
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
```

Use this workaround only when your environment actually has the relevant SQLite compatibility problem.

---

# 🛡️ Git & Security

The following files and directories are intentionally ignored:

```text
.venv/
.env
chroma_db/
*.pdf
__pycache__/
.vscode/
```

This prevents:

* Virtual environments from being uploaded
* API keys from being exposed
* Large local vector databases from entering the repository
* Local/private documents from being published accidentally

---

# 🧪 Technologies Used

| Technology                | Purpose                         |
| ------------------------- | ------------------------------- |
| **Python**                | Core programming language       |
| **Streamlit**             | Interactive web interface       |
| **LangChain**             | RAG application framework       |
| **ChromaDB**              | Vector database                 |
| **Hugging Face**          | Embeddings                      |
| **Sentence Transformers** | Semantic embeddings             |
| **Mistral AI**            | Large Language Model            |
| **Google Gemini**         | Embeddings / AI experimentation |
| **PyPDF**                 | PDF document processing         |
| **Git & GitHub**          | Version control                 |

---

# 📈 Future Improvements

Possible future enhancements for PageMind AI include:

* Multi-document conversations
* Conversation memory
* Source/page citations
* Streaming LLM responses
* Better document previews
* OCR for scanned PDFs
* Hybrid search
* Reranking models
* Authentication and user accounts
* Cloud-based vector storage
* Support for DOCX, TXT, CSV and web pages
* Multiple LLM provider selection
* RAG evaluation and retrieval metrics

---

# 🎓 Learning Goals

This project is designed to provide practical experience with:

* Retrieval-Augmented Generation
* Vector databases
* Embedding models
* Semantic search
* Document processing
* LangChain
* LLM application development
* Information retrieval
* Streamlit application development

---

# 👨‍💻 Author

**Niraj Singh**

Building AI-powered applications and exploring:

```text
Artificial Intelligence
Machine Learning
Generative AI
RAG Systems
LLM Applications
Data Science
```

---

# ⭐ Support the Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.

**Repository:**
https://github.com/ImNirajsingh/PageMind-AI

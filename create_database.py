#   RAG Syastem - PHASE

#load pdf 
#split into chunks 
#create the embeddings 
#store into chroma 




from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv


load_dotenv()

data = PyPDFLoader("C:\\Educational Coding\\RAG System\\document loaders\\deep-learning-material-dept-ece-ase-blr-1.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)


from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

print("=" * 50)
print("✅ Vector database created successfully!")
print(f"📄 Total chunks: {len(chunks)}")
print("=" * 50)


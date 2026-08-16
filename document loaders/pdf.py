from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter



data = PyPDFLoader("C:\\Educational Coding\\RAG System\\document loaders\\GRU.pdf")
docs = data.load()


text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

chunks = text_splitter.split_documents(docs)
print(chunks[0].page_content)
print(chunks[1].page_content)
print(chunks[2].page_content)
print(chunks[3].page_content)
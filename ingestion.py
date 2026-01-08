# Import the Pinecone library
import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv, dotenv_values
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()
config = dotenv_values(".env")

folder_path = "data/photography_ingestion/"
pdf_files = []

for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        pdf_files.append(os.path.join(folder_path, filename))
        
docs = [PyPDFLoader(file).load() for file in pdf_files]

docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=200
)
doc_splits = text_splitter.split_documents(docs_list)
print(doc_splits[0])


""" 
Checking token counts (optional)
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

total_tokens = 0

for doc in doc_splits:
    total_tokens += count_tokens(doc.page_content)

print("Total tokens:", total_tokens)
"""


# Initialize a Pinecone client with your API key
pc = Pinecone(api_key=config["PINECONE_API_KEY"])

# Create a dense index with integrated embedding
index_name = "photocoach-ai-index"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(index_name)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
vector_store.add_documents(doc_splits)

# results = vector_store.similarity_search(
#     "How do i reduce noise in my photos?",
#     k=2)

# for result in results:
#     print(f"* {result.page_content} [{result.metadata}]")
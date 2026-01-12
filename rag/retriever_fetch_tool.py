from dotenv import load_dotenv, dotenv_values
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.tools import tool

load_dotenv()
config = dotenv_values(".env")

pc = Pinecone(api_key=config["PINECONE_API_KEY"])

index_name = "photocoach-ai-index"
index = pc.Index(index_name)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

@tool
def retrieve_photography_tips(query: str) -> list[str]:
    """Search and return information about photography tips from the books ingested."""
    docs = retriever.invoke(query)
    # return "\n\n".join([doc.page_content for doc in docs])
    return [doc.page_content for doc in docs]

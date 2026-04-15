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
# MMR (Maximal Marginal Relevance) fetches 20 candidates then returns the 5 most
# relevant *and* diverse chunks, avoiding redundant passages from the same source.
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.6},
)

@tool
def retrieve_photography_tips(query: str) -> list[str]:
    """Search and return information about photography tips from the books ingested."""
    docs = retriever.invoke(query)
    # return "\n\n".join([doc.page_content for doc in docs])
    return [doc.page_content for doc in docs]

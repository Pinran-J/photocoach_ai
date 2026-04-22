import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.tools import tool
from sentence_transformers import CrossEncoder

load_dotenv()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index_name = "photocoach-ai-index"
index = pc.Index(index_name)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# MMR fetches 40 candidates then returns 10 diverse ones for the reranker to work with.
# Increasing fetch_k gives the reranker a wider pool to pick from.
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10, "fetch_k": 40, "lambda_mult": 0.6},
)

# CrossEncoder reranker: scores each (query, chunk) pair directly rather than
# relying on embedding similarity alone — much more precise relevance signal.
# Model is ~80MB, loaded once at startup.
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


@tool
def retrieve_photography_tips(query: str) -> list[str]:
    """Search and return photography tips from ingested books and articles."""
    docs = retriever.invoke(query)           # MMR → 10 diverse candidates
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)         # score each (query, chunk) pair
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc.page_content for _, doc in ranked[:5]]   # return top 5 by relevance

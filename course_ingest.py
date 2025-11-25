from pydantic import BaseModel, Field, ValidationError
from typing import Annotated
from bs4 import BeautifulSoup
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.retrievers import ContextualCompressionRetriever
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv(override=True)

def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator="\n")
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = "college_seeker"
COURSE_VECTORS_COLL = "course_vectors"
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "course-index-vector")
VECTOR_QUERY_K = int(os.getenv("VECTOR_QUERY_K", "12"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "6"))
#RERANK_MODEL = os.getenv("RERANK_MODEL", "ms-marco-MultiBERT-L-12")

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]
collection = db[COURSE_VECTORS_COLL]
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name=VECTOR_INDEX_NAME,
)
vector_retriever = vector_store.as_retriever()
flashrank_compressor = FlashrankRerank()
compression_retriever = ContextualCompressionRetriever(
    base_retriever=vector_retriever,
    base_compressor=flashrank_compressor,
)

llm = init_chat_model("google_genai:gemini-2.5-flash")
from langchain.agents import create_agent

from langchain_community.document_loaders import RecursiveUrlLoader

#non functional function for now
def ingest_college_data(url):
    
    loader=RecursiveUrlLoader(url,extractor=bs4_extractor,prevent_outside=True)

    documents = loader.load()


    # print("Documents loaded:", len(documents))
    # for doc in documents:
    #     print("Docs Metadata:", doc.metadata)
    #     print("Document content:", doc.page_content[:200])  # Print first 200 characters of each document

    splitter=RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs= splitter.split_documents(documents)
    print("Documents after splitting:", len(docs))

    
    vector_store.create_vector_search_index(dimensions=768)

    vector_store.add_documents(docs)



# results = vector_store.similarity_search(
#     "B.Tech in Computer Engineering details",
# )
# print(results)

from langchain.agents.middleware import dynamic_prompt, ModelRequest

tavily_search_tool = TavilySearch(
    max_results=5,
    topic="general",
)

@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    retrieved_docs = vector_retriever.invoke(last_query) if last_query else []

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    system_message = (
        "You are College‑Seeker Assistant. Use ONLY the retrieved documents below as authoritative context.\n"
        "- For tool calling and in General keep in mind to keep an Indian Context and search the web for the Institute's info if not available\n"
        "- Prefer official college pages, admissions pages, government / accreditation sources and cite each factual claim with the source URL.\n"
        "- Return a concise direct answer (1–3 sentences). Then, when helpful, include a short 'Details' section with bullets for: Program, Degree, Duration, Fees, Eligibility, Important links, Contact.\n"
        "PLEASE FETCH DATA FROM THE RETRIEVED DOCUMENTS FIRST. THIS IS A MAJOR PRIORITY. You can also user Web Search to provide more context from the retrieved information."
        "\n\n"
        f"{docs_content}"
    )

    return system_message


agent = create_agent(llm, tools=[tavily_search_tool], middleware=[prompt_with_context])

def process_course_query(query: str):
    """Process a course query through the agent.
    
    Args:
        query: The query string to process (typically from student ingest agent)
    
    Returns:
        The final agent response
    """
    if not query or not query.strip():
        raise ValueError("Course query cannot be empty.")

    response = agent.invoke({"messages": [{"role": "user", "content": query}]})

    messages = response.get("messages", [])
    if not messages:
        raise RuntimeError("Agent returned an empty response.")

    ai_message = messages[-1]
    # LangChain returns AIMessage objects that may carry string or chunked structured content.
    if hasattr(ai_message, "content"):
        content = ai_message.content
        if isinstance(content, list):
            # Flatten text chunks coming from models that return structured content.
            text_parts = []
            for chunk in content:
                if isinstance(chunk, str):
                    text_parts.append(chunk)
                elif isinstance(chunk, dict):
                    text_parts.append(chunk.get("text", ""))
            return "\n".join(part for part in text_parts if part)
        if isinstance(content, str):
            return content
    if isinstance(ai_message, dict):
        content = ai_message.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for chunk in content:
                if isinstance(chunk, str):
                    text_parts.append(chunk)
                elif isinstance(chunk, dict):
                    text_parts.append(chunk.get("text", ""))
            return "\n".join(part for part in text_parts if part)
        return str(content)
    return str(ai_message)


def get_reranked_courses(query: str, limit: int | None = None) -> list[dict]:
    """Return top course documents ranked via contextual compression."""
    if not query or not query.strip():
        #print("nothing here")
        return []

    docs = compression_retriever.invoke(query.strip())
    if limit is not None:
        docs = docs[:limit]

    hits: list[dict] = []
    for doc in docs:
        score = doc.metadata.get("relevance_score")
        try:
            score_val = float(score) if score is not None else None
        except (TypeError, ValueError):
            score_val = None

        hits.append(
            {
                "course_id": doc.metadata.get("course_id"),
                "url": doc.metadata.get("url"),
                "snippet": doc.page_content,
                "score": score_val,
            }
        )
    return hits
    
    
if __name__ == "__main__":
    # Default query if running standalone
    user_input = "I am a looking for an b.tech degree" 
    #print(process_course_query(user_input))
    print(get_reranked_courses(user_input))
    
#ingest_college_data()

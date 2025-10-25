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
from pymongo import MongoClient
import os
from dotenv import load_dotenv
load_dotenv(override=True)

def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator="\n")
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


client=MongoClient(os.getenv("MONGODB_URI"))
db="college_seeker"
collection_name="college_data_raw"
collection=client[db][collection_name]
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store= MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name="college-index-vectorstores",
    )

llm = init_chat_model("google_genai:gemini-2.5-flash-lite")
from langchain.agents import create_agent

from langchain_community.document_loaders import RecursiveUrlLoader

#non functional function for now
def ingest_college_data():
    
    loader=RecursiveUrlLoader("https://www.tcetmumbai.in/deptCompEngineering-home.html",extractor=bs4_extractor,prevent_outside=True)

    documents = loader.load()


    # print("Documents loaded:", len(documents))
    # for doc in documents:
    #     print("Docs Metadata:", doc.metadata)
    #     print("Document content:", doc.page_content[:200])  # Print first 200 characters of each document

    splitter=RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
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
    retrieved_docs = vector_store.similarity_search(last_query)

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    system_message = system_message = (
        "You are College‑Seeker Assistant. Use ONLY the retrieved documents below as authoritative context.\n"
        "- If the answer is not present in the context, reply: \"I don't know.\" Do not hallucinate.\n"
        "- Prefer official college pages, admissions pages, government / accreditation sources and cite each factual claim with the source URL.\n"
        "- Return a concise direct answer (1–3 sentences). Then, when helpful, include a short 'Details' section with bullets for: Program, Degree, Duration, Fees, Eligibility, Important links, Contact.\n"
        "- When the caller requests machine-readable output, return JSON with keys: title, answer, details, sources (list of {title, url}), confidence (0.0-1.0).\n"
        "- Always include a 'sources' list containing the page title and URL for each cited statement.\n\n"
        f"{docs_content}"
    )

    return system_message


agent = create_agent(llm, tools=[tavily_search_tool], middleware=[prompt_with_context])

user_input = "I have completed my 12th with PCM and want to pursue a B.Tech in Computer Engineering. Can you suggest some good colleges in Mumbai along with their admission criteria, fees, and important links?"

for step in agent.stream(
    {"messages": user_input},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
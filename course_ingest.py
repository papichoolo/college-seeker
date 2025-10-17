from langchain_community.document_loaders import PyPDFLoader
uri = "mongodb+srv://tmber:<password>@testcluster.wyfpg0v.mongodb.net/?retryWrites=true&w=majority&appName=testcluster"

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from uuid import uuid4
from dotenv import load_dotenv
load_dotenv(override=True)
from langchain import hub

from typing import Optional

from pydantic import BaseModel, Field
from langchain_core.documents import Document
from typing_extensions import List, Annotated, TypedDict

# COURSE SCHEMA
class CourseDetailsDict(TypedDict):
    courseName: Annotated[str, "Course title as printed in the brochure."]
    courseDuration: Annotated[int, "Nominal duration in years (integer)."]
    courseCoreSubjects: Annotated[List[str], "List of mandatory/core subjects."]
    courseElectives: Annotated[List[str], "List of elective subjects (empty if none listed)."]
    courseSpecialisations: Annotated[List[str], "Tracks/streams/specialisations offered."]
    coursePrereqs: Annotated[List[str], "Admission prerequisites and eligibility points."]


class InstituteSchemaRequired(TypedDict):
    institution: Annotated[str, "Official name of the institution."]
    degrees: Annotated[List[str], "All degrees/programs mentioned in the brochure."]
    courseDetails: Annotated[CourseDetailsDict, "Primary course block captured from the brochure."]
    features: Annotated[str, "One-paragraph summary of facilities, curriculum design, and distinctive features."]


class InstituteSchemaOptional(TypedDict, total=False):
    placementRecords: Annotated[str, "Notable placement statistics, recruiters, or salary figures if provided."]
    awards: Annotated[str, "Awards, accreditations, and notable recognitions if provided."]


class InstituteSchema(InstituteSchemaRequired, InstituteSchemaOptional):
    """Final schema: required + optional fields combined."""


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
# N.B. for non-US LangSmith endpoints, you may need to specify
# api_url="https://api.smith.langchain.com" in hub.pull.
prompt = hub.pull("rlm/rag-prompt")

import getpass
import os

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

from langchain.chat_models import init_chat_model

llm = init_chat_model("gemini-2.5-pro", model_provider="google_genai")

structured_llm= llm.with_structured_output(InstituteSchema)

example_messages = prompt.invoke(
    {"context": "(context goes here)", "question": "(question goes here)"}
).to_messages()

assert len(example_messages) == 1
print(example_messages[0].content)

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = structured_llm.invoke(messages)
    return {"answer": response}

from langgraph.graph import START, StateGraph

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

client = MongoClient(uri)
DB_NAME = "college_seeker"
COLLECTION_NAME = "rawcourse_collection"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "rawcourse-index-vectorstores"
MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]
vector_store = MongoDBAtlasVectorSearch(
            collection=MONGODB_COLLECTION,
            embedding=embeddings,
            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
            relevance_score_fn="cosine",
        )

def ingest_course_pdf(file_path):
    try:
        # Load the PDF file
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Split the documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        
        # Initialize MongoDB client and vector store
        
        
        # Create vector search index on the collection
        vector_store.create_vector_search_index(dimensions=768)
        
        # Add documents to the vector store
        vector_store.add_documents(docs, ids=[str(uuid4()) for _ in range(len(docs))])
        
        return {"status": "success", "message": f"Processed {len(docs)} document chunks."}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
    
def query_course_pdf() -> Optional[InstituteSchema]:
    result=graph.invoke({"question": "Convert the ENTIRE DOCUMENT to a JSON object with fields: institution, degrees, courseDetails, features, placementRecords, awards."})
    return result['answer']



print(query_course_pdf())
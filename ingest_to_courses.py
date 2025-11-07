# ingest_courses_llm.py
import os, re, json
from datetime import datetime, timezone
from pymongo import MongoClient, ReplaceOne
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import trafilatura
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.documents import Document
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from test_chain import build_extractor, CourseSchema
from pydantic import ValidationError
from typing import Optional

# --- Config ---
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = "college_seeker"
COURSES_COLL = "courses"                 # structured, user-facing
COURSE_VECTORS_COLL = "course_vectors"   # for retrieval/ranking
VECTOR_INDEX_NAME = "course-index-vector"
EMB_MODEL = "sentence-transformers/all-mpnet-base-v2"

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]
courses = db[COURSES_COLL]
course_vectors = db[COURSE_VECTORS_COLL]

embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
vs = MongoDBAtlasVectorSearch(
    collection=course_vectors,
    embedding=embeddings,
    index_name=VECTOR_INDEX_NAME,
)

def bs4_extractor(html: str) -> str:
    """
    Extracts clean text from raw HTML.
    - First tries `trafilatura` for article-like, main-body content.
    - Falls back to BeautifulSoup's text extraction if Trafilatura fails.
    - Cleans up redundant blank lines and whitespace.
    """
    # Try trafilatura first
    extracted = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        favor_recall=True
    )

    if extracted and extracted.strip():
        # Clean and normalize whitespace
        text = re.sub(r"\n\s*\n+", "\n\n", extracted)
        return text.strip()

    # Fallback to BeautifulSoup if trafilatura didn't find main content
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator="\n")
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()

def course_id_from_struct(s: CourseSchema) -> str:
    # simple deterministic slug
    def slug(x: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", x.lower()).strip("-")
    return f"{slug(s.institution_name)}__{slug(s.course_name)}"

def summarize_for_embedding(s: CourseSchema) -> str:
    # compact semantic profile for ranking
    parts = [
        f"{s.institution_name} – {s.course_name}",
        f"Level: {s.degree_level}, Mode: {', '.join(s.delivery_mode or [])}",
        f"Duration: {s.duration_value or ''} {s.duration_unit or ''}",
        f"Fee (INR): {s.fee_value_inr or ''}",
        f"Eligibility: {s.eligibility_min_qualification or ''} {s.eligibility_min_percentage or ''}",
        f"Topics: {', '.join(s.topics or [])}",
        f"Accreditation: {', '.join(s.accreditation or [])}",
    ]
    return " | ".join(p.strip() for p in parts if p and str(p).strip())


def _try_parse_course_schema(candidate) -> Optional[CourseSchema]:
    """Attempt to coerce various agent payload shapes into CourseSchema."""
    if candidate is None:
        return None
    if isinstance(candidate, CourseSchema):
        return candidate
    if hasattr(candidate, "model_dump"):
        try:
            return CourseSchema.model_validate(candidate.model_dump())
        except ValidationError:
            return None
    if isinstance(candidate, dict):
        try:
            return CourseSchema.model_validate(candidate)
        except ValidationError:
            return None
    if isinstance(candidate, str):
        text = candidate.strip()
        if not text:
            return None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        return _try_parse_course_schema(data)
    if isinstance(candidate, list):
        for item in candidate:
            parsed = _try_parse_course_schema(item)
            if parsed is not None:
                return parsed
        return None
    if hasattr(candidate, "__dict__"):
        return _try_parse_course_schema(candidate.__dict__)
    return None


def _extract_message_content(message):
    """Return the most relevant content field from a LangChain message object."""
    if message is None:
        return None
    if hasattr(message, "content"):
        return message.content
    if isinstance(message, dict):
        return message.get("content")
    return message


def _parse_agent_response(response) -> CourseSchema:
    """Parse the dynamic agent response back into CourseSchema."""
    messages = None
    if isinstance(response, dict):
        messages = response.get("messages")
    elif hasattr(response, "get"):
        try:
            messages = response.get("messages")
        except Exception:
            messages = None
    if messages and not isinstance(messages, list):
        messages = [messages]
    if not messages and isinstance(response, list):
        messages = response
    if not messages:
        raise RuntimeError("Agent returned an unexpected payload with no messages.")

    # Iterate from last to first to prioritize final AI message.
    for message in reversed(messages):
        content = _extract_message_content(message)
        parsed = _try_parse_course_schema(content)
        if parsed is not None:
            return parsed
        if hasattr(message, "additional_kwargs"):
            parsed = _try_parse_course_schema(getattr(message, "additional_kwargs"))
            if parsed is not None:
                return parsed
    raise RuntimeError("Could not parse CourseSchema from agent response.")


def invoke_extractor_for_page(extractor, page_url: str, page_text: str) -> CourseSchema:
    """Invoke the dynamic agent and coerce its reply into a CourseSchema instance."""
    if not page_text or not page_text.strip():
        raise ValueError(f"No page text available for {page_url}")

    user_prompt = (
        "Extract structured course information using the CourseSchema. Use ONLY the provided page_text.\n"
        "If a field is missing, return null. Normalize Indian currency amounts to absolute INR values.\n"
        f"URL: {page_url}\n\n"
        f"PAGE_TEXT:\n{page_text}"
    )
    response = extractor.invoke(
        {"messages": [{"role": "user", "content": user_prompt}]}
    )
    course = _parse_agent_response(response)
    if not course.source_url:
        course.source_url = page_url
    return course

def ingest_root(url: str, max_depth: int = 1):
    # 1) Crawl
    loader = RecursiveUrlLoader(
        url,
        extractor=bs4_extractor,
        max_depth=max_depth,
        prevent_outside=True,
        check_response_status=True,
    )
    raw_docs = loader.load()  # returns List[Document]
    # 2) Split (keeps metadata.source)
    splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=200)
    chunks = splitter.split_documents(raw_docs)

    @dynamic_prompt
    def prompt_with_context(request: ModelRequest) -> str:
        """Inject context into state messages."""
        last_query = request.state["messages"][-1].text
        retrieved_docs = vs.similarity_search(last_query)

        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

        system_message = system_message = (
            "You are College‑Seeker Assistant. Use ONLY the retrieved documents below as authoritative context.\n"
            "- For tool calling and in General keep in mind to keep an Indian Context\n"
            "- Prefer official college pages, admissions pages, government / accreditation sources and cite each factual claim with the source URL.\n"
            "- Return a concise direct answer (1–3 sentences). Then, when helpful, include a short 'Details' section with bullets for: Program, Degree, Duration, Fees, Eligibility, Important links, Contact.\n"
            "PLEASE FETCH DATA FROM THE RETRIEVED DOCUMENTS FIRST. THIS IS A MAJOR PRIORITY. You can also user Web Search to provide more context from the retrieved information."
            "\n\n"
            f"{docs_content}"
        )

        return system_message
    extractor = build_extractor(prompt_fn=prompt_with_context)

    # 4) Run per URL (merge all chunks from same page before calling LLM)
    by_url = {}
    for d in chunks:
        by_url.setdefault(d.metadata.get("source", url), []).append(d.page_content)

    upserts = []
    vec_docs = []
    for page_url, texts in by_url.items():
        page_text = "\n\n".join(texts)[:100_000]  # keep prompt size sane
        print(page_text)
        structured = invoke_extractor_for_page(extractor, page_url, page_text)
        print(structured)
        # compute deterministic id & doc
        _id = course_id_from_struct(structured)
        doc = {
            "_id": _id,
            "course_id": _id,
            "institution": {"name": structured.institution_name},
            "course_name": structured.course_name,
            "degree_level": structured.degree_level,
            "delivery_mode": structured.delivery_mode,
            "duration": ({"value": structured.duration_value, "unit": structured.duration_unit}
                         if structured.duration_value else None),
            "fee": ({"value": structured.fee_value_inr, "currency": "INR"}
                    if structured.fee_value_inr else None),
            "eligibility": {
                "minimum_qualification": structured.eligibility_min_qualification,
                "minimum_percentage": structured.eligibility_min_percentage,
            },
            "accreditation": structured.accreditation,
            "topics": structured.topics,
            "syllabus_summary": structured.syllabus_summary,
            "learning_outcomes": structured.learning_outcomes,
            "career_paths": structured.career_paths,
            "tags": list({*(structured.topics or []), *(structured.accreditation or [])}),
            "source": {"url": structured.source_url, "retrieved_date": datetime.now(timezone.utc)},
            "last_updated": datetime.now(timezone.utc),
        }
        upserts.append(ReplaceOne({"_id": _id}, doc, upsert=True))

        # 5) Vectorize a compact profile for personalization/ranking
        profile = summarize_for_embedding(structured)
        vec_docs.append(Document(
            page_content=profile,
            metadata={"course_id": _id, "url": page_url}
        ))

    if upserts:
        courses.bulk_write(upserts)
    if vec_docs:
        # ensure/declare the Atlas Vector index (idempotent):
        vs.create_vector_search_index(dimensions=768)
        vs.add_documents(vec_docs)

if __name__ == "__main__":
    ingest_root("https://nitw.ac.in/ap", max_depth=1)

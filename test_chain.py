# models_and_chain.py
from datetime import datetime
from dotenv import load_dotenv
load_dotenv(override=True)
from typing import List, Optional
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from pydantic import BaseModel, Field  # safe for LC v1
from langchain_core.runnables import RunnableSerializable

# ----- Pydantic schema the LLM must fill -----
class CourseSchema(BaseModel):
    institution_name: str = Field(..., description="Full official name of the institution")
    course_name: str = Field(..., description="Exact title as shown on page, e.g., 'B.Tech in Computer Engineering'")
    degree_level: Optional[str] = Field(None, description="Undergraduate, Postgraduate, Diploma/Certificate, Doctorate")
    delivery_mode: Optional[List[str]] = Field(None, description="Any of: Full-time, Part-time, Online, Hybrid")
    duration_value: Optional[float] = Field(None, description="Numeric duration")
    duration_unit: Optional[str] = Field(None, description="years | months | semesters")
    fee_value_inr: Optional[float] = Field(None, description="Approx total/annual fee in INR (normalize lakh/crore to absolute)")
    eligibility_min_qualification: Optional[str] = None
    eligibility_min_percentage: Optional[float] = None
    accreditation: Optional[List[str]] = Field(None, description="Eg. NAAC, AICTE, UGC, NBA if present")
    topics: Optional[List[str]] = Field(None, description="Key subjects/topics covered")
    syllabus_summary: Optional[str] = None
    learning_outcomes: Optional[List[str]] = None
    career_paths: Optional[List[str]] = None
    source_url: str


# this function can be used to web search GROUNDED Results by providing it a URL
# def extract_web_data(url):
#     #url can then be string formatted in a better way to get more information
#     #https://www.iitk.ac.in/department courses listing, schools, departments
#     #Use the Tavily API to search with better result
#     return 0

tavily_search_tool = TavilySearch(
    max_results=5,
    topic="general",
)


def build_extractor(
    prompt_fn,
    model_name: str = "google_genai:gemini-2.5-flash",
) -> RunnableSerializable:
    """
    Build an agent that can call Tavily but always responds with CourseSchema data.

    The base chat model stays unwrapped so the compiler can still decide when to
    issue tool calls; schema enforcement now comes from `response_format`.
    """
    llm = init_chat_model(model_name)
    middleware = [prompt_fn] if prompt_fn is not None else []
    system_prompt = (
        "You extract structured course information. "
        "Use ONLY provided context (retrieved passages, user text, tool outputs). "
        "If a field is unknown, return null and cite the source URL."
    )
    agent = create_agent(
        model=llm,
        tools=[tavily_search_tool],
        system_prompt=system_prompt,
        middleware=middleware,
        response_format=CourseSchema,
    )
    return agent

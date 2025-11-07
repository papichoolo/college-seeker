# models_and_chain.py
from datetime import datetime
from dotenv import load_dotenv
load_dotenv(override=True)
from typing import List, Optional
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
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

def build_extractor(prompt_fn,model_name: str = "google_genai:gemini-2.5-flash") -> RunnableSerializable:
    llm = init_chat_model(model_name)  # unified init in LC v1
    # Ask LLM to ONLY use provided text
    # prompt = ChatPromptTemplate.from_messages([
    #     ("system",
    #      "You extract structured course data from web page text. "
    #      "Use ONLY the provided page_text—no outside knowledge. "
    #      "If a field is missing, return null. Normalize Indian fees to absolute INR."),
    #     ("human",
    #      "URL: {url}\n\n"
    #      "PAGE_TEXT (may be truncated or noisy):\n{page_text}\n\n"
    #      #"WEB_RESULTS: {web_results}"
    #      "Return the result strictly as the provided schema.")
    # ])
    structured_llm = llm.with_structured_output(CourseSchema)
    agent = create_agent(structured_llm, tools=[tavily_search_tool], middleware=[prompt_fn])
    return agent
    
    # llm.bind_tools(tavily_search_tool)
    
    # # LC v1 structured outputs
    # return prompt | structured_llm 


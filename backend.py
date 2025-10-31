# The backend python file which will use FastAPI to create the API endpoints

# from pymongo.mongo_client import MongoClient
# from pymongo.server_api import ServerApi

# uri = "mongodb+srv://tmber:@testcluster.wyfpg0v.mongodb.net/?retryWrites=true&w=majority&appName=testcluster"

# # Create a new client and connect to the server
# client = MongoClient(uri, server_api=ServerApi('1'))

# # Send a ping to confirm a successful connection
# try:
#     client.admin.command('ping')
#     print("Pinged your deployment. You successfully connected to MongoDB!")
# except Exception as e:
#     print(e)


# from langchain_huggingface import HuggingFaceEmbeddings

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# if embeddings:
#     print("Embeddings model loaded successfully!")
from student_ingest import ingest_student_pdf, ingest_student_web, make_student_analysis
from course_ingest import ingest_college_data, process_course_query
import os

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(
    title="College Seeker API",
    description="API for student profile analysis and course recommendations",
    version="1.0.0"
)

# Pydantic models for request/response
class StudentInfo(BaseModel):
    name: str = Field(..., description="Student's full name")
    
class StudentAnalysisRequest(BaseModel):
    student_info: StudentInfo
    question: str = Field(
        default="Answer with which Degree, Course Specialization to consider keeping in mind my Academic, Co-curricular and Extracurricular Profile.",
        description="Question to ask the student analysis agent"
    )

class CourseQueryRequest(BaseModel):
    query: str = Field(..., description="Query for course recommendations")

class IntegratedFlowRequest(BaseModel):
    student_name: str = Field(..., description="Student's full name")
    question: str = Field(
        default="What type of College Should I apply to with this information?",
        description="Question to ask the student analysis agent"
    )

@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {
        "message": "College Seeker API is running",
        "version": "1.0.0",
        "endpoints": {
            "upload_student_pdf": "/uploadfile/",
            "upload_student_link": "/uploadlink/",
            "upload_course_pdf": "/uploadcourse/",
            "student_analysis": "/api/student/analyze",
            "course_recommendations": "/api/courses/recommend",
            "integrated_flow": "/api/analyze-and-recommend"
        }
    }

@app.post("/uploadfile/")
def create_upload_file(file: UploadFile = File(...)):
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are accepted.")
    try:
        """file_location = f"files/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())"""
        # Call the ingest function to process and store the PDF content
        #store the file as a temp file and pass the path to the ingest function
        
        # Ensure temp directory exists
        os.makedirs("temp", exist_ok=True)
        
        temp_file_path = f"temp/{file.filename}"
        with open(temp_file_path, "wb+") as temp_file:
            temp_file.write(file.file.read())
        result = ingest_student_pdf(temp_file_path)
        return {"info": f"file '{file.filename}' processed successfully", "details": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the file: {str(e)}")
    


@app.post("/uploadlink/")
def create_upload_link(link: str):
    try:
        # Call the ingest function to process and store the content from the link
        result = ingest_student_web(link)
        return {"info": f"link '{link}' processed successfully", "details": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the link: {str(e)}")


@app.post("/uploadcourse/")
def create_upload_course(file: UploadFile = File(...)):
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are accepted.")
    try:
        """file_location = f"files/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())"""
        # Call the ingest function to process and store the PDF content
        #store the file as a temp file and pass the path to the ingest function
        
        # Ensure temp directory exists
        os.makedirs("temp", exist_ok=True)
        
        temp_file_path = f"temp/{file.filename}"
        with open(temp_file_path, "wb+") as temp_file:
            temp_file.write(file.file.read())
        result = ingest_college_data(temp_file_path)
        return {"info": f"file '{file.filename}' processed successfully", "details": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the file: {str(e)}")


@app.post("/api/student/analyze")
async def analyze_student_profile(request: StudentAnalysisRequest):
    """
    Analyze a student's profile based on their information stored in the database.
    
    - **student_info**: Contains the student's name
    - **question**: The question to ask about the student's profile
    
    Returns the analysis from the student agent.
    """
    try:
        # Prepare the query for the student agent
        query = {
            "messages": [{"role": "user", "content": request.question}],
            "student_info": {"name": request.student_info.name}
        }
        
        # Get student analysis
        analysis = make_student_analysis(query)
        
        return {
            "student_name": request.student_info.name,
            "question": request.question,
            "analysis": analysis
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing student profile: {str(e)}")


@app.post("/api/courses/recommend")
async def get_course_recommendations(request: CourseQueryRequest):
    """
    Get course recommendations based on a query.
    
    - **query**: The search query for courses (e.g., student's preferences, field of study)
    
    Returns course recommendations from the course agent.
    """
    try:
        # Get course recommendations
        recommendations = process_course_query(request.query)
        
        return {
            "query": request.query,
            "recommendations": recommendations
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting course recommendations: {str(e)}")


@app.post("/api/analyze-and-recommend")
async def integrated_analysis_and_recommendation(request: IntegratedFlowRequest):
    """
    Complete integrated flow: Analyze student profile and get course recommendations.
    
    This endpoint:
    1. Analyzes the student's profile based on their name
    2. Uses the analysis output as input for course recommendations
    3. Returns both the analysis and recommendations
    
    - **student_name**: The student's full name (must exist in database)
    - **question**: Question to ask the student analysis agent
    """
    try:
        # Step 1: Analyze student profile
        student_query = {
            "messages": [{"role": "user", "content": request.question}],
            "student_info": {"name": request.student_name}
        }
        
        student_analysis = make_student_analysis(student_query)
        
        # Step 2: Use student analysis as input for course recommendations
        course_recommendations = process_course_query(student_analysis)
        
        return {
            "student_name": request.student_name,
            "question": request.question,
            "student_analysis": student_analysis,
            "course_recommendations": course_recommendations
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in integrated flow: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
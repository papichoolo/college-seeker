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
from student_ingest import ingest_student_pdf, ingest_student_web
from course_ingest import ingest_course_pdf
import os

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

app= FastAPI()

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
        result = ingest_course_pdf(temp_file_path)
        return {"info": f"file '{file.filename}' processed successfully", "details": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
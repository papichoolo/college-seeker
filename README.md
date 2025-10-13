# AI Course Matching System - Proof of Concept

This is a proof of concept implementation of an AI-powered course matching system that ingests student profiles from resumes and web sources, then recommends relevant courses based on their background and interests.

## Architecture Overview

The system consists of:
- **Profile Ingestion**: Resume parsing and web profile adapters
- **Course Catalog**: Vector database of courses for similarity matching
- **Recommendation Engine**: LangChain-based retrieval and ranking system

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

3. Run the application:
```bash
streamlit run app.py
```

## Project Structure

```
├── app.py                  # Main Streamlit application
├── src/
│   ├── __init__.py
│   ├── profile_ingestion/
│   │   ├── __init__.py
│   │   ├── resume_parser.py
│   │   └── url_adapters.py
│   ├── catalog_ingestion/
│   │   ├── __init__.py
│   │   └── course_db.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── query_builder.py
│   │   ├── hybrid_retriever.py
│   │   └── recommender.py
│   └── database/
│       ├── __init__.py
│       ├── profiles_db.py
│       └── vector_store.py
├── data/
│   ├── sample_resumes/
│   └── sample_courses/
└── requirements.txt
```
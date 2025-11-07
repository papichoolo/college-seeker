"""
College Seeker - Streamlit Frontend
A user-friendly interface for getting personalized college and course recommendations
"""

import os
import re
from datetime import timedelta
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True)

import pandas as pd
import requests
import streamlit as st
from pymongo import MongoClient

# =========================
# API / Page Configuration
# =========================
API_BASE_URL = "http://localhost:8080"

st.set_page_config(
    page_title="College Seeker",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Backend API Helpers
# =========================
def check_api_status():
    """Check if the backend API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def upload_pdf(file):
    """Upload a PDF file to the backend"""
    try:
        files = {"file": (file.name, file, "application/pdf")}
        response = requests.post(f"{API_BASE_URL}/uploadfile/", files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error uploading file: {str(e)}")
        return None


def upload_link(url):
    """Upload a profile link to the backend"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/uploadlink/",
            params={"link": url}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error uploading link: {str(e)}")
        return None


def get_recommendations(student_name, question=None):
    """Get integrated analysis and course recommendations"""
    try:
        payload = {"student_name": student_name}
        if question:
            payload["question"] = question

        response = requests.post(
            f"{API_BASE_URL}/api/analyze-and-recommend",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return None


# =========================
# MongoDB Helpers (Courses)
# =========================
@st.cache_resource(show_spinner=False)
def get_mongo_client():
    """Create a cached MongoDB client."""
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI env var not set.")
    return MongoClient(uri, uuidRepresentation="standard")


def get_collection():
    """Return the course_vectors collection."""
    client = get_mongo_client()
    db = client["college_seeker"]
    return db["course_vectors"]


def _default_projection():
    """Projection that hides large / noisy fields like embeddings."""
    return {
        "_id": 0,
        "text": 1,
        "course_id": 1,
        "url": 1,
        # If you later add structured fields, include them here.
        # "degree": 1, "duration_years": 1, ...
        "last_updated": 1,
    }


@st.cache_data(ttl=timedelta(minutes=2), show_spinner=False)
def fetch_courses(filters: dict, page: int, page_size: int, sort=[("text", 1)]):
    """
    Fetch paginated course docs (excluding embeddings).
    Returns (rows, total_count).
    """
    col = get_collection()
    query = filters or {}
    total = col.count_documents(query)
    cursor = (
        col.find(query, _default_projection())
           .skip((page - 1) * page_size)
           .limit(page_size)
           .sort(sort)
    )
    rows = list(cursor)
    return rows, total


def parse_course_text(doc: dict) -> dict:
    """
    Extracts structured fields from a single 'text' string like:
    "Institute — Course | Level: X, Mode: Y | Duration: Z years | Fee (INR): ... | Eligibility: ... | Topics: ... | Accreditation: ..."

    Returns a dict with clean columns for display.
    """
    text = doc.get("text", "") or ""
    url = doc.get("url", "") or ""
    course_id = doc.get("course_id", "") or ""
    last_updated = doc.get("last_updated", "")

    # Split by pipes and the long dash
    parts = [p.strip() for p in re.split(r"\s*\|\s*|—", text) if p and p.strip()]

    result = {
        "Institute": "",
        "Course": "",
        "Level": "",
        "Mode": "",
        "Duration": "",
        "Fee (INR)": "",
        "Eligibility": "",
        "Accreditation": "",
        "Course ID": course_id,
        "URL": url,
        "Last Updated": last_updated,
        "Raw": text,  # keep original for search/debug if needed
    }

    # Institute and Course (first two chunks if present)
    if len(parts) >= 2:
        result["Institute"] = parts[0]
        result["Course"] = parts[1]

    # Parse remaining labeled chunks
    for part in parts[2:]:
        # Support both "Level: Undergraduate, Mode: Full-time" and similar
        if part.lower().startswith("level:"):
            result["Level"] = part.split(":", 1)[1].strip()
        elif part.lower().startswith("mode:"):
            result["Mode"] = part.split(":", 1)[1].strip()
        elif part.lower().startswith("duration:"):
            result["Duration"] = part.split(":", 1)[1].strip()
        elif part.lower().startswith("fee"):
            # Handles "Fee (INR): ..." or "Fee:"
            result["Fee (INR)"] = part.split(":", 1)[1].strip() if ":" in part else part
        elif part.lower().startswith("eligibility:"):
            result["Eligibility"] = part.split(":", 1)[1].strip()
        elif part.lower().startswith("accreditation:"):
            result["Accreditation"] = part.split(":", 1)[1].strip()

    return result


# =========================
# Page Functions
# =========================
def main():
    # Header
    st.title("🎓 College Seeker")
    st.markdown("#### Find Your Perfect College & Course Match")

    # Check API status (non-blocking for the Mongo page)
    api_ok = check_api_status()

    # Initialize session state
    if "profile_uploaded" not in st.session_state:
        st.session_state.profile_uploaded = False
    if "student_name" not in st.session_state:
        st.session_state.student_name = None
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = None

    # Sidebar
    with st.sidebar:
        st.title("📋 Navigation")
        st.markdown("---")
        page = st.radio(
            "Choose a step:",
            [
                "1️⃣ Upload Profile",
                "2️⃣ Get Recommendations",
                "3️⃣ About",
                "Generated Recommendations",
                "4️⃣ Browse Courses",
            ],
            index=0 if not st.session_state.profile_uploaded else 1
        )
        st.markdown("---")
        st.markdown("### Status")
        if st.session_state.profile_uploaded and st.session_state.student_name:
            st.success(f"✅ Profile uploaded for:\n**{st.session_state.student_name}**")
        elif st.session_state.student_name:
            st.info(f"🔍 Using existing data for:\n**{st.session_state.student_name}**")
        else:
            st.info("⏳ No profile uploaded yet")

        st.markdown("---")
        st.markdown("### Quick Actions")
        if st.button("🔄 Reset Session"):
            st.session_state.clear()
            st.rerun()

        st.markdown("---")
        st.markdown("### Backend API")
        if api_ok:
            st.success("✅ Backend API is running")
        else:
            st.warning("⚠️ Backend API not reachable")

    # Routing
    if page == "1️⃣ Upload Profile":
        if not api_ok:
            st.error("⚠️ Backend API is not running. Please start the backend server first.")
            st.code("python backend.py", language="bash")
            st.stop()
        show_upload_page()
    elif page == "2️⃣ Get Recommendations":
        if not api_ok:
            st.error("⚠️ Backend API is not running. Please start the backend server first.")
            st.code("python backend.py", language="bash")
            st.stop()
        show_recommendations_page()
    elif page == "Generated Recommendations" and st.session_state.recommendations:
        display_recommendations(st.session_state.recommendations)
    elif page == "4️⃣ Browse Courses":
        show_courses_page()
    else:
        show_about_page(api_ok)


def show_upload_page():
    """Page for uploading student profile"""
    st.header("📤 Upload Your Profile")
    st.markdown("Choose how you'd like to share your profile with us:")

    tab1, tab2 = st.tabs(["📄 Upload Resume (PDF)", "🔗 Profile Link"])

    with tab1:
        with st.container():
            st.subheader("Upload Your Resume")
            st.markdown("Upload your resume in PDF format. We'll analyze your skills, education, and experience.")
            uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
            col1, _ = st.columns([3, 1])
            with col1:
                student_name_pdf = st.text_input(
                    "Your Full Name",
                    key="name_pdf",
                    placeholder="e.g., Sourav Dutta",
                    help="Enter your full name as it appears in your resume"
                )

            if uploaded_file and student_name_pdf:
                if st.button("📤 Upload Resume", type="primary", key="upload_pdf_btn"):
                    with st.spinner("Uploading and processing your resume..."):
                        result = upload_pdf(uploaded_file)
                        if result:
                            st.session_state.profile_uploaded = True
                            st.session_state.student_name = student_name_pdf
                            st.balloons()
                            st.success(f"✅ {result.get('info', 'Success')}")
                            st.info(f"📊 {result.get('details', '')}")
                            st.markdown("---")
                            st.markdown("### ✨ Next Step")
                            st.markdown("Go to **'Get Recommendations'** to see your personalized course suggestions!")

    with tab2:
        with st.container():
            st.subheader("Share Your Profile Link")
            st.markdown("Provide a link to your LinkedIn, portfolio, or any online profile.")
            profile_url = st.text_input(
                "Profile URL",
                placeholder="https://linkedin.com/in/yourprofile or your portfolio website"
            )
            col1, _ = st.columns([3, 1])
            with col1:
                student_name_link = st.text_input(
                    "Your Full Name",
                    key="name_link",
                    placeholder="e.g., Sourav Dutta"
                )

            if profile_url and student_name_link:
                if st.button("🔗 Submit Profile Link", type="primary", key="upload_link_btn"):
                    with st.spinner("Fetching and processing your profile..."):
                        result = upload_link(profile_url)
                        if result:
                            st.session_state.profile_uploaded = True
                            st.session_state.student_name = student_name_link
                            st.balloons()
                            st.success(f"✅ {result.get('info', 'Success')}")
                            st.info(f"📊 {result.get('details', '')}")
                            st.markdown("---")
                            st.markdown("### ✨ Next Step")
                            st.markdown("Go to **'Get Recommendations'** to see your personalized course suggestions!")


def show_recommendations_page():
    """Page for getting course recommendations"""
    st.header("🎯 Get Your Personalized Recommendations")

    if not st.session_state.profile_uploaded:
        st.info("ℹ️ Uploading a profile helps the AI, but you can still request recommendations with just your name.")

    with st.container():
        existing_name = st.session_state.student_name if st.session_state.student_name else ""
        name_input = st.text_input(
            "Your Full Name",
            value=existing_name,
            key="recommendations_name",
            placeholder="e.g., Sourav Dutta",
        )
        provided_name = name_input.strip()
        previous_name = (existing_name or "").strip()

        if provided_name:
            if provided_name != previous_name and st.session_state.get("recommendations"):
                st.session_state.recommendations = None
            st.session_state.student_name = provided_name
            st.markdown(f"### Hello, **{provided_name}**! 👋")
            st.markdown("Let's find the perfect college and courses for you.")
        else:
            st.session_state.student_name = None
            if st.session_state.get("recommendations"):
                st.session_state.recommendations = None
            st.markdown("### Hello! 👋")
            st.info("Enter your full name to run the recommendation engine.")

        with st.expander("🔧 Customize Your Query (Optional)"):
            custom_question = st.text_area(
                "Ask a specific question about your college preferences:",
                placeholder="What type of college should I apply to with this information?"
            )

        _, col2, _ = st.columns([1, 2, 1])
        with col2:
            if st.button("✨ Get My Recommendations", type="primary", use_container_width=True):
                if not provided_name:
                    st.error("Please enter your full name before requesting recommendations.")
                else:
                    with st.spinner("🔍 Analyzing your profile and finding the best matches..."):
                        question = custom_question.strip() if (custom_question or "").strip() else None
                        recommendations = get_recommendations(provided_name, question)
                        if recommendations:
                            st.session_state.recommendations = recommendations

    if st.session_state.get("recommendations"):
        st.success("Recommendations are available for you to view in the 'Generated Recommendations' Tab")


def display_recommendations(recommendations: dict):
    """Display the recommendations in a nice format"""
    st.markdown("## 📊 Your Results")

    # Student Analysis
    st.markdown("### 🎓 Profile Analysis")
    analysis_text = recommendations.get('student_analysis', 'No analysis available')
    st.info(analysis_text)

    st.markdown("---")

    # Course Recommendations
    st.markdown("### 🏫 Recommended Colleges & Courses")
    course_recs = recommendations.get('course_recommendations', 'No recommendations available')
    st.success(course_recs)

    # Actions
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("🔄 Get New Recommendations"):
            st.session_state.recommendations = None
            st.rerun()

    with col2:
        if st.button("📥 Prepare Download"):
            # Create downloadable text file
            result_text = f"""
COLLEGE SEEKER - RECOMMENDATIONS
================================

Student: {st.session_state.student_name}
Date: {st.session_state.recommendations.get('timestamp', 'N/A')}

PROFILE ANALYSIS:
{recommendations.get('student_analysis', 'N/A')}

COURSE RECOMMENDATIONS:
{recommendations.get('course_recommendations', 'N/A')}
"""
            st.download_button(
                label="Download as Text File",
                data=result_text,
                file_name=f"recommendations_{(st.session_state.student_name or 'student').replace(' ', '_')}.txt",
                mime="text/plain"
            )

    with col3:
        if st.button("📤 Upload New Profile"):
            st.session_state.clear()
            st.rerun()


def show_courses_page():
    """Browse courses stored in MongoDB -> college_seeker.course_vectors"""
    st.header("📚 Browse Courses (MongoDB)")
    st.caption("Data source: `college_seeker.course_vectors`")

    # ── Filters
    with st.container():
        col_a, col_b, col_c = st.columns([2, 2, 1])
        with col_a:
            kw = st.text_input(
                "🔍 Search keyword",
                placeholder="Search Institute / Course / Eligibility / Accreditation…"
            ).strip()
        with col_b:
            id_kw = st.text_input(
                "Filter by Course ID (optional)",
                placeholder="e.g., thakur-college-of-engineering-technology__b-e-computer-engineering"
            ).strip()
        with col_c:
            page_size = st.selectbox("Rows per page", [10, 20, 50, 100], index=1)

    # Build Mongo query
    filters = {}
    or_clauses = []

    if kw:
        or_clauses.extend([
            {"text": {"$regex": kw, "$options": "i"}},
            {"url": {"$regex": kw, "$options": "i"}},
        ])
    if id_kw:
        or_clauses.append({"course_id": {"$regex": id_kw, "$options": "i"}})

    if or_clauses:
        filters = {"$or": or_clauses}

    # Pagination controls
    col_l, _, col_r = st.columns([1, 2, 1])
    with col_l:
        page = st.number_input("Page", min_value=1, value=1, step=1)
    with col_r:
        refresh = st.button("🔄 Refresh", use_container_width=True)

    # Fetch data
    try:
        with st.spinner("Loading courses…"):
            docs, total = fetch_courses(filters, page=int(page), page_size=int(page_size))

        if not docs:
            st.info("No courses found with the current filters.")
            return

        # Parse text -> structured columns
        parsed_rows = [parse_course_text(doc) for doc in docs]
        df = pd.DataFrame(parsed_rows)

        # Reorder columns
        preferred_cols = [
            "Institute", "Course", "Level", "Mode", "Duration",
            "Fee (INR)", "Eligibility", "Accreditation",
            "Course ID", "URL", "Last Updated"
        ]
        existing_order = [c for c in preferred_cols if c in df.columns]
        rest_cols = [c for c in df.columns if c not in existing_order]
        df = df[existing_order + rest_cols]

        # Render table (use LinkColumn for URLs)
        max_page = max(1, (total + int(page_size) - 1) // int(page_size))
        st.markdown(f"**Total matches:** {total} | **Page:** {page} of {max_page} | **Page size:** {page_size}")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "URL": st.column_config.LinkColumn("URL", display_text="Visit Site"),
            },
        )

        # Download current page
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download this page as CSV",
            data=csv,
            file_name=f"courses_page_{page}.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error loading courses: {e}")
        st.stop()


def show_about_page(api_ok: bool):
    """About page with information"""
    st.header("ℹ️ About College Seeker")

    st.markdown("""
    ### 🎯 What is College Seeker?

    College Seeker is an AI-powered platform that helps students find the perfect college and courses 
    based on their academic background, skills, and career aspirations.

    ### 🚀 How It Works

    1. **Upload Your Profile**: Share your resume or profile link with us  
    2. **AI Analysis**: Our intelligent system analyzes your:
       - Educational background
       - Skills and expertise
       - Career interests
       - Academic achievements
    3. **Get Recommendations**: Receive personalized suggestions for:
       - Best-fit colleges
       - Relevant courses
       - Admission criteria
       - Important links and resources

    ### ✨ Features

    - 🤖 **AI-Powered Analysis**: Advanced AI agents analyze your complete profile
    - 🎓 **Personalized Recommendations**: Tailored suggestions to your background
    - 🔍 **Comprehensive Search**: Browse college/course data from MongoDB
    - 🌐 **Web Integration**: Supports both PDF uploads and online profile links

    ### 🛠️ Technology Stack

    - **Frontend**: Streamlit
    - **Backend**: FastAPI
    - **AI Models**: Google Gemini, LangChain Agents
    - **Database**: MongoDB Atlas with Vector Search
    - **Embeddings**: HuggingFace Sentence Transformers
    """)

    with st.expander("🔧 System Status"):
        st.markdown(f"**Backend API:** {'✅ Connected' if api_ok else '⚠️ Not reachable'}")
        st.markdown(f"**API URL:** `{API_BASE_URL}`")

        if api_ok:
            try:
                response = requests.get(f"{API_BASE_URL}/")
                if response.status_code == 200:
                    data = response.json()
                    st.json(data)
            except Exception:
                st.error("Could not fetch API details")


# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    main()

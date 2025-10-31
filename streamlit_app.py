"""
College Seeker - Streamlit Frontend
A user-friendly interface for getting personalized college and course recommendations
"""

import streamlit as st
import requests
import os
from pathlib import Path

# API Configuration
API_BASE_URL = "http://localhost:8080"

# Page configuration
st.set_page_config(
    page_title="College Seeker",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .step-card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .info-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    </style>
    """, unsafe_allow_html=True)

def check_api_status():
    """Check if the backend API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        return response.status_code == 200
    except:
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
        payload = {
            "student_name": student_name
        }
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

def main():
    # Header
    st.markdown('<div class="main-header">🎓 College Seeker</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Find Your Perfect College & Course Match</div>', unsafe_allow_html=True)
    
    # Check API status
    if not check_api_status():
        st.error("⚠️ Backend API is not running. Please start the backend server first.")
        st.code("python backend.py", language="bash")
        st.stop()
    
    # Initialize session state
    if 'profile_uploaded' not in st.session_state:
        st.session_state.profile_uploaded = False
    if 'student_name' not in st.session_state:
        st.session_state.student_name = None
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    
    # Sidebar for navigation
    with st.sidebar:
        st.title("📋 Navigation")
        st.markdown("---")
        
        page = st.radio(
            "Choose a step:",
            ["1️⃣ Upload Profile", "2️⃣ Get Recommendations", "3️⃣ About"],
            index=0 if not st.session_state.profile_uploaded else 1
        )
        
        st.markdown("---")
        st.markdown("### Status")
        if st.session_state.profile_uploaded:
            st.success(f"✅ Profile uploaded for:\n**{st.session_state.student_name}**")
        else:
            st.info("⏳ No profile uploaded yet")
        
        st.markdown("---")
        st.markdown("### Quick Actions")
        if st.button("🔄 Reset Session"):
            st.session_state.clear()
            st.rerun()
    
    # Main content based on selected page
    if page == "1️⃣ Upload Profile":
        show_upload_page()
    elif page == "2️⃣ Get Recommendations":
        show_recommendations_page()
    else:
        show_about_page()

def show_upload_page():
    """Page for uploading student profile"""
    st.header("📤 Upload Your Profile")
    st.markdown("Choose how you'd like to share your profile with us:")
    
    # Create tabs for different upload methods
    tab1, tab2 = st.tabs(["📄 Upload Resume (PDF)", "🔗 Profile Link"])
    
    with tab1:
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.subheader("Upload Your Resume")
        st.markdown("Upload your resume in PDF format. We'll analyze your skills, education, and experience.")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload your resume or CV in PDF format"
        )
        
        col1, col2 = st.columns([3, 1])
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
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.subheader("Share Your Profile Link")
        st.markdown("Provide a link to your LinkedIn, portfolio, or any online profile.")
        
        profile_url = st.text_input(
            "Profile URL",
            placeholder="https://linkedin.com/in/yourprofile or your portfolio website",
            help="Enter the complete URL of your online profile"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            student_name_link = st.text_input(
                "Your Full Name",
                key="name_link",
                placeholder="e.g., Sourav Dutta",
                help="Enter your full name"
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
        
        st.markdown('</div>', unsafe_allow_html=True)

def show_recommendations_page():
    """Page for getting course recommendations"""
    st.header("🎯 Get Your Personalized Recommendations")

    if not st.session_state.profile_uploaded:
        st.info("ℹ️ Uploading a profile helps the AI, but you can still request recommendations with just your name.")

    st.markdown('<div class="step-card">', unsafe_allow_html=True)

    existing_name = st.session_state.student_name if st.session_state.student_name else ""
    name_input = st.text_input(
        "Your Full Name",
        value=existing_name,
        key="recommendations_name",
        placeholder="e.g., Sourav Dutta",
        help="Enter the name associated with your uploaded profile"
    )
    provided_name = name_input.strip()
    previous_name = existing_name.strip() if isinstance(existing_name, str) else ""

    if provided_name:
        if provided_name != previous_name and st.session_state.recommendations:
            st.session_state.recommendations = None
        st.session_state.student_name = provided_name
        st.markdown(f"### Hello, **{provided_name}**! 👋")
        st.markdown("Let's find the perfect college and courses for you.")
    else:
        st.session_state.student_name = None
        if st.session_state.recommendations:
            st.session_state.recommendations = None
        st.markdown("### Hello! 👋")
        st.info("Enter your full name to run the recommendation engine.")

    # Custom question (optional)
    with st.expander("🔧 Customize Your Query (Optional)"):
        custom_question = st.text_area(
            "Ask a specific question about your college preferences:",
            placeholder="What type of College Should I apply to with this information?",
            help="Leave blank to use the default question"
        )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("✨ Get My Recommendations", type="primary", use_container_width=True):
            if not provided_name:
                st.error("Please enter your full name before requesting recommendations.")
            else:
                with st.spinner("🔍 Analyzing your profile and finding the best matches..."):
                    question = custom_question if custom_question.strip() else None
                    recommendations = get_recommendations(
                        provided_name,
                        question
                    )

                    if recommendations:
                        st.session_state.recommendations = recommendations

    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display recommendations if available
    if st.session_state.recommendations:
        st.markdown("---")
        display_recommendations(st.session_state.recommendations)

def display_recommendations(recommendations):
    """Display the recommendations in a nice format"""
    st.markdown("## 📊 Your Results")
    
    # Student Analysis
    st.markdown("### 🎓 Profile Analysis")
    analysis_container = st.container()
    with analysis_container:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown(recommendations.get('student_analysis', 'No analysis available'))
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Course Recommendations
    st.markdown("### 🏫 Recommended Colleges & Courses")
    rec_container = st.container()
    with rec_container:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        course_recs = recommendations.get('course_recommendations', 'No recommendations available')
        st.markdown(course_recs)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🔄 Get New Recommendations"):
            st.session_state.recommendations = None
            st.rerun()
    
    with col2:
        if st.button("📥 Download Results"):
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
                file_name=f"recommendations_{st.session_state.student_name.replace(' ', '_')}.txt",
                mime="text/plain"
            )
    
    with col3:
        if st.button("📤 Upload New Profile"):
            st.session_state.clear()
            st.rerun()

def show_about_page():
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
    - 🎓 **Personalized Recommendations**: Get suggestions tailored to your unique background
    - 🔍 **Comprehensive Search**: Access information about colleges, courses, fees, and eligibility
    - 📊 **Detailed Insights**: Understand why specific colleges and courses match your profile
    - 🌐 **Web Integration**: Supports both PDF uploads and online profile links
    
    ### 🛠️ Technology Stack
    
    - **Frontend**: Streamlit
    - **Backend**: FastAPI
    - **AI Models**: Google Gemini, LangChain Agents
    - **Database**: MongoDB Atlas with Vector Search
    - **Embeddings**: HuggingFace Sentence Transformers
    
    ### 📞 Support
    
    For questions or support, please contact our team.
    
    ---
    
    **Version 1.0.0** | Made with ❤️ for students seeking their perfect college match
    """)
    
    # System status
    with st.expander("🔧 System Status"):
        st.markdown("**Backend API:** ✅ Connected")
        st.markdown(f"**API URL:** `{API_BASE_URL}`")
        
        try:
            response = requests.get(f"{API_BASE_URL}/")
            if response.status_code == 200:
                data = response.json()
                st.json(data)
        except:
            st.error("Could not fetch API details")

if __name__ == "__main__":
    main()

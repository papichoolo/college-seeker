"""
AI Course Matching System - Main Application
Streamlit web interface for the course recommendation system
"""

import streamlit as st
import os
import sys
from dotenv import load_dotenv
import logging

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.profile_ingestion.resume_parser import ResumeParser
from src.profile_ingestion.url_adapters import URLProfileAdapter, MultiProfileExtractor
from src.catalog_ingestion.course_db import CourseDatabase, create_sample_courses
from src.database.profiles_db import ProfilesDatabase
from src.retrieval.recommender import CourseRecommender

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Course Matching System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_system():
    """Initialize all system components"""
    try:
        # Initialize databases
        profiles_db = ProfilesDatabase("profiles.db")
        course_db = CourseDatabase("./chroma_db")
        
        # Check if course database is empty and populate with sample data
        stats = course_db.get_database_stats()
        if stats.get('total_courses', 0) == 0:
            st.info("Initializing course database with sample courses...")
            sample_courses = create_sample_courses()
            course_db.add_courses_from_list(sample_courses)
            st.success(f"Added {len(sample_courses)} sample courses to database")
        
        # Initialize parsers and recommender
        resume_parser = ResumeParser()
        url_adapter = URLProfileAdapter()
        multi_extractor = MultiProfileExtractor()
        recommender = CourseRecommender(profiles_db, course_db)
        
        return {
            'profiles_db': profiles_db,
            'course_db': course_db,
            'resume_parser': resume_parser,
            'url_adapter': url_adapter,
            'multi_extractor': multi_extractor,
            'recommender': recommender
        }
        
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        return None

def main():
    """Main application function"""
    
    # Title and description
    st.title("🎓 AI Course Matching System")
    st.markdown("""
    **Intelligent course recommendations based on your profile and goals**
    
    This system analyzes your resume and online profiles to recommend the most relevant courses for your learning journey.
    """)
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        st.error("⚠️ Please set your OpenAI API key in the .env file")
        st.info("Create a .env file with: OPENAI_API_KEY=your_api_key_here")
        return
    
    # Initialize system
    with st.spinner("Initializing AI Course Matching System..."):
        system = initialize_system()
    
    if not system:
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Profile Creation", "Course Recommendations", "Browse Courses", "System Stats"]
    )
    
    if page == "Profile Creation":
        profile_creation_page(system)
    elif page == "Course Recommendations":
        recommendations_page(system)
    elif page == "Browse Courses":
        browse_courses_page(system)
    elif page == "System Stats":
        system_stats_page(system)

def profile_creation_page(system):
    """Page for creating student profiles"""
    st.header("📝 Create Your Profile")
    
    tab1, tab2, tab3 = st.tabs(["Resume Upload", "URL Profile", "Manual Entry"])
    
    with tab1:
        st.subheader("Upload Resume (PDF)")
        
        uploaded_file = st.file_uploader("Choose a PDF resume", type="pdf")
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                with st.spinner("Parsing resume..."):
                    parsed_data = system['resume_parser'].parse_resume_file(temp_path)
                
                st.success("Resume parsed successfully!")
                
                # Display parsed information
                st.subheader("Extracted Information")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.text_input("Name", value=parsed_data.get('name', ''), key="resume_name")
                    st.text_input("Email", value=parsed_data.get('email', ''), key="resume_email")
                    st.text_input("Phone", value=parsed_data.get('phone', ''), key="resume_phone")
                
                with col2:
                    st.text_area("Skills", value=parsed_data.get('skills', ''), key="resume_skills")
                    st.text_area("Education", value=parsed_data.get('education', ''), key="resume_education")
                
                st.text_area("Experience", value=parsed_data.get('experience', ''), key="resume_experience")
                st.text_area("Summary", value=parsed_data.get('summary', ''), key="resume_summary")
                
                if st.button("Save Profile from Resume"):
                    # Create profile summary
                    profile_summary = system['resume_parser'].create_profile_summary(parsed_data)
                    parsed_data['profile_summary'] = profile_summary
                    parsed_data['source_type'] = 'resume'
                    
                    # Save to database
                    profile_id = system['profiles_db'].insert_profile(parsed_data)
                    st.success(f"Profile saved with ID: {profile_id}")
                
                # Clean up temp file
                os.remove(temp_path)
                
            except Exception as e:
                st.error(f"Error parsing resume: {str(e)}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
    
    with tab2:
        st.subheader("Extract from URL Profile")
        
        url_input = st.text_input("Enter profile URL (LinkedIn, GitHub, Portfolio, etc.)")
        
        if url_input and st.button("Extract Profile"):
            try:
                with st.spinner("Extracting profile information..."):
                    profile_data = system['url_adapter'].extract_profile_from_url(url_input)
                
                st.success("Profile extracted successfully!")
                
                # Display extracted information
                st.subheader("Extracted Information")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.text_input("Name", value=profile_data.get('name', ''), key="url_name")
                    st.text_input("Title", value=profile_data.get('title', ''), key="url_title")
                    st.text_input("Company", value=profile_data.get('company', ''), key="url_company")
                
                with col2:
                    st.text_area("Skills", value=profile_data.get('skills', ''), key="url_skills")
                    st.text_area("Education", value=profile_data.get('education', ''), key="url_education")
                
                st.text_area("Experience", value=profile_data.get('experience', ''), key="url_experience")
                st.text_area("Projects", value=profile_data.get('projects', ''), key="url_projects")
                st.text_area("Summary", value=profile_data.get('summary', ''), key="url_summary")
                
                if st.button("Save Profile from URL"):
                    # Create profile summary
                    profile_summary = system['url_adapter'].create_profile_summary(profile_data)
                    profile_data['profile_summary'] = profile_summary
                    
                    # Save to database
                    profile_id = system['profiles_db'].insert_profile(profile_data)
                    st.success(f"Profile saved with ID: {profile_id}")
                
            except Exception as e:
                st.error(f"Error extracting profile: {str(e)}")
    
    with tab3:
        st.subheader("Manual Profile Entry")
        
        with st.form("manual_profile"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Full Name")
                email = st.text_input("Email")
                phone = st.text_input("Phone")
            
            with col2:
                skills = st.text_area("Technical Skills (comma-separated)")
                education = st.text_area("Education Background")
            
            experience = st.text_area("Work Experience")
            summary = st.text_area("Professional Summary")
            
            if st.form_submit_button("Save Manual Profile"):
                if name and email:
                    profile_data = {
                        'name': name,
                        'email': email,
                        'phone': phone,
                        'skills': skills,
                        'education': education,
                        'experience': experience,
                        'summary': summary,
                        'source_type': 'manual'
                    }
                    
                    # Create profile summary
                    profile_summary = f"Name: {name}\nSkills: {skills}\nEducation: {education}\nExperience: {experience}\nSummary: {summary}"
                    profile_data['profile_summary'] = profile_summary
                    
                    try:
                        profile_id = system['profiles_db'].insert_profile(profile_data)
                        st.success(f"Profile saved with ID: {profile_id}")
                    except Exception as e:
                        st.error(f"Error saving profile: {str(e)}")
                else:
                    st.error("Please provide at least name and email")

def recommendations_page(system):
    """Page for getting course recommendations"""
    st.header("🎯 Get Course Recommendations")
    
    # Profile selection
    profiles = system['profiles_db'].get_all_profiles(limit=50)
    
    if not profiles:
        st.warning("No profiles found. Please create a profile first.")
        return
    
    # Create profile selection dropdown
    profile_options = {f"{p['name']} ({p['email']})": p['id'] for p in profiles}
    selected_profile_name = st.selectbox("Select a profile:", list(profile_options.keys()))
    selected_profile_id = profile_options[selected_profile_name]
    
    # Recommendation settings
    col1, col2 = st.columns(2)
    
    with col1:
        max_courses = st.slider("Number of recommendations", 1, 20, 10)
    
    with col2:
        level_filter = st.multiselect(
            "Course levels:", 
            ["Beginner", "Intermediate", "Advanced"],
            help="Leave empty for all levels"
        )
    
    # Additional filters
    with st.expander("Advanced Filters"):
        dept_filter = st.multiselect(
            "Departments:",
            ["Computer Science", "Data Science", "Mathematics", "Business"],
            help="Leave empty for all departments"
        )
        
        category_filter = st.multiselect(
            "Categories:",
            ["Programming", "Analytics", "Web Development", "Artificial Intelligence", "Security"],
            help="Leave empty for all categories"
        )
    
    if st.button("Get Recommendations", type="primary"):
        # Prepare filters
        filters = {}
        if level_filter:
            filters['level'] = level_filter
        if dept_filter:
            filters['department'] = dept_filter
        if category_filter:
            filters['category'] = category_filter
        
        try:
            with st.spinner("Generating personalized recommendations..."):
                recommendations = system['recommender'].get_recommendations(
                    selected_profile_id, 
                    max_courses=max_courses, 
                    filters=filters
                )
            
            # Display results
            st.success(f"Found {len(recommendations['recommendations'])} recommendations!")
            
            # Show profile analysis
            with st.expander("Profile Analysis"):
                analysis = recommendations['analysis']
                st.write("**Skill Gaps:**", analysis.get('skill_gaps', 'Not specified'))
                st.write("**Career Goals:**", analysis.get('career_goals', 'Not specified'))
                st.write("**Learning Level:**", analysis.get('learning_level', 'Not specified'))
                st.write("**Search Query:**", analysis.get('search_query', 'Not specified'))
            
            # Display recommendations
            st.subheader("📚 Recommended Courses")
            
            for i, rec in enumerate(recommendations['recommendations'], 1):
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.markdown(f"**{i}. {rec['title']}**")
                        st.write(f"*{rec['code']} - {rec['department']}*")
                        st.write(rec['explanation'])
                    
                    with col2:
                        st.metric("Level", rec['level'])
                        st.metric("Credits", rec['credits'])
                    
                    with col3:
                        st.metric("Match Score", f"{rec['final_score']:.2f}")
                        if rec['instructor']:
                            st.write(f"**Instructor:** {rec['instructor']}")
                    
                    st.divider()
            
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")

def browse_courses_page(system):
    """Page for browsing available courses"""
    st.header("📖 Browse Courses")
    
    # Search functionality
    search_query = st.text_input("Search courses:", placeholder="Enter keywords, skills, or topics")
    
    col1, col2 = st.columns(2)
    with col1:
        search_limit = st.slider("Number of results", 1, 20, 10)
    
    if search_query:
        try:
            results = system['course_db'].search_similar_courses(search_query, k=search_limit)
            
            st.subheader(f"Search Results ({len(results)} courses)")
            
            for i, result in enumerate(results, 1):
                metadata = result['metadata']
                
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**{i}. {metadata.get('title', 'Unknown Course')}**")
                        st.write(f"*{metadata.get('code', '')} - {metadata.get('department', '')}*")
                        
                        # Show course content preview
                        content_preview = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
                        st.write(content_preview)
                    
                    with col2:
                        st.write(f"**Level:** {metadata.get('level', 'N/A')}")
                        st.write(f"**Credits:** {metadata.get('credits', 'N/A')}")
                        st.write(f"**Instructor:** {metadata.get('instructor', 'N/A')}")
                        st.write(f"**Similarity:** {result['similarity_score']:.3f}")
                    
                    st.divider()
                    
        except Exception as e:
            st.error(f"Error searching courses: {str(e)}")
    else:
        st.info("Enter a search query to find relevant courses")

def system_stats_page(system):
    """Page showing system statistics"""
    st.header("📊 System Statistics")
    
    try:
        # Get database stats
        profile_stats = system['profiles_db'].get_database_stats()
        course_stats = system['course_db'].get_database_stats()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Profile Database")
            st.metric("Total Profiles", profile_stats.get('total_profiles', 0))
            st.metric("Total Sources", profile_stats.get('total_sources', 0))
            st.write(f"**Last Updated:** {profile_stats.get('last_updated', 'Never')}")
            st.write(f"**Database Path:** {profile_stats.get('database_path', 'N/A')}")
        
        with col2:
            st.subheader("Course Database")
            st.metric("Total Courses", course_stats.get('total_courses', 0))
            st.write(f"**Persist Directory:** {course_stats.get('persist_directory', 'N/A')}")
        
        # Recent profiles
        st.subheader("Recent Profiles")
        recent_profiles = system['profiles_db'].get_all_profiles(limit=10)
        
        if recent_profiles:
            for profile in recent_profiles:
                with st.container():
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.write(f"**{profile['name']}**")
                    
                    with col2:
                        st.write(profile['email'])
                    
                    with col3:
                        st.write(profile['source_type'])
                    
                    st.divider()
        else:
            st.info("No profiles found")
        
    except Exception as e:
        st.error(f"Error getting system stats: {str(e)}")

if __name__ == "__main__":
    main()
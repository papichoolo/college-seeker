"""
Integrated flow connecting student ingest agent to course ingest agent.
The output from student analysis becomes the input query for course recommendations.
"""

from student_ingest import make_student_analysis
from course_ingest import process_course_query

def run_integrated_flow(student_name: str, student_question: str = "Answer with which Degree, Course Specialization to consider keeping in mind my Academic, Co-curricular and Extracurricular Profile."):
    """
    Run the complete flow: student analysis -> course recommendations
    
    Args:
        student_name: The name of the student whose profile to analyze
        student_question: The question to ask the student agent
    
    Returns:
        Dictionary with both student analysis and course recommendations
    """
    print("=" * 80)
    print("STEP 1: Analyzing Student Profile")
    print("=" * 80)
    
    # Get student analysis from student ingest agent
    student_analysis = make_student_analysis(
        {
            "messages": [{"role": "user", "content": student_question}],
            "student_info": {"name": student_name}
        }
    )
    
    print(f"\nStudent Analysis Result:\n{student_analysis}\n")
    
    print("=" * 80)
    print("STEP 2: Finding Course Recommendations")
    print("=" * 80)
    
    # Use student analysis as input for course ingest agent
    course_recommendations = process_course_query(student_analysis)
    
    print("\n" + "=" * 80)
    print("INTEGRATION COMPLETE")
    print("=" * 80)
    
    return {
        "student_analysis": student_analysis,
        "course_recommendations": course_recommendations
    }


if __name__ == "__main__":
    # Example usage
    result = run_integrated_flow(
        student_name="Sourav Dutta",
    )
    
    print("\n\nFINAL RESULTS:")
    print("-" * 80)
    print(f"Student Profile Analysis: {result['student_analysis']}")
    print("-" * 80)
    print(f"Course Recommendations: {result['course_recommendations']}")

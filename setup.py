"""
Setup Script for AI Course Matching System
"""

import os
import sys
import subprocess
import json

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def create_env_file():
    """Create .env file if it doesn't exist"""
    if not os.path.exists('.env'):
        print("Creating .env file...")
        with open('.env', 'w') as f:
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n")
            f.write("CHROMA_PERSIST_DIRECTORY=./chroma_db\n")
        print("✅ .env file created. Please add your OpenAI API key!")
        return False
    else:
        print("✅ .env file already exists")
        return True

def setup_directories():
    """Create necessary directories"""
    directories = [
        "chroma_db",
        "data/sample_resumes",
        "logs"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    try:
        import langchain
        import chromadb
        import streamlit
        import pandas
        import numpy
        import PyPDF2
        import requests
        from bs4 import BeautifulSoup
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main setup function"""
    print("🎓 AI Course Matching System Setup")
    print("=" * 40)
    
    # Install requirements
    if not install_requirements():
        return
    
    # Test imports
    if not test_imports():
        return
    
    # Create directories
    setup_directories()
    
    # Create .env file
    env_exists = create_env_file()
    
    print("\n" + "=" * 40)
    print("🚀 Setup Complete!")
    print("=" * 40)
    
    if not env_exists:
        print("⚠️  Please add your OpenAI API key to the .env file")
        print("   Then run: streamlit run app.py")
    else:
        print("✅ Ready to run: streamlit run app.py")
    
    print("\nNext steps:")
    print("1. Add your OpenAI API key to .env file")
    print("2. Run: streamlit run app.py")
    print("3. Create profiles and get course recommendations!")

if __name__ == "__main__":
    main()
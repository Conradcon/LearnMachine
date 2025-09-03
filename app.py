import streamlit as st
import sys
import os

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from classification_exercise import ClassificationExercise
from dataset_analysis import DatasetAnalysis
from model_fitting import ModelFitting

def main():
    st.set_page_config(
        page_title="ML Education Hub",
        page_icon="https://www.ucd.ie/fileupload/ucd-logo.png",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'user_progress' not in st.session_state:
        st.session_state.user_progress = {
            'classification_scores': [],
            'exercises_completed': 0,
            'current_exercise': None
        }
    
    # Sidebar navigation
    st.sidebar.title("ML Education Hub")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Choose Learning Module:",
        [
            "🏠 Home",
            "🎯 Classification Exercises", 
            "📊 Dataset Analysis",
            "🔧 Model Fitting Lab"
        ]
    )
    
    # Progress display
    with st.sidebar:
        st.markdown("### 📈 Your Progress")
        progress_col1, progress_col2 = st.columns(2)
        with progress_col1:
            st.metric("Exercises Completed", st.session_state.user_progress['exercises_completed'])
        with progress_col2:
            avg_score = sum(st.session_state.user_progress['classification_scores']) / max(1, len(st.session_state.user_progress['classification_scores']))
            st.metric("Average Score", f"{avg_score:.1f}%")
    
    # Main content area
    if page == "🏠 Home":
        show_home()
    elif page == "🎯 Classification Exercises":
        classification_app = ClassificationExercise()
        classification_app.run()
    elif page == "📊 Dataset Analysis":
        analysis_app = DatasetAnalysis()
        analysis_app.run()
    elif page == "🔧 Model Fitting Lab":
        fitting_app = ModelFitting()
        fitting_app.run()

def show_home():
    st.title("Machine Learning Education Hub")
    st.markdown("### Welcome to Interactive ML Learning!")
    
    
    st.markdown("""
    This educational platform helps you understand core machine learning concepts through 
    interactive exercises and hands-on activities. Choose from the modules below to start learning:
    """)
    
    # Feature cards
    st.markdown("### 🎓 Learning Modules")
    
    module_col1, module_col2, module_col3 = st.columns(3)
    
    with module_col1:
        st.markdown("""
        **🎯 Classification Exercises**
        - Pattern recognition with visual examples
        - Supervised learning fundamentals
        - Interactive card-based activities
        - Immediate feedback and scoring
        """)
    with module_col2:    
        st.markdown("""
        **🔧 Model Fitting Lab**
        - Hands-on regression and classification
        - Parameter tuning exercises
        - Visual model performance analysis
        - Compare different algorithms
        """)
    
    with module_col3:
        st.markdown("""
        **📊 Dataset Analysis**
        - Explore real-world datasets
        - Data visualization techniques
        - Feature analysis and correlation
        - Data preprocessing steps
        """)
    
    # Recent activity
    st.markdown("---")
    st.markdown("### 📈 Learning Statistics")
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    with stats_col1:
        st.metric("Total Exercises", st.session_state.user_progress['exercises_completed'])
    
    with stats_col2:
        total_scores = len(st.session_state.user_progress['classification_scores'])
        st.metric("Attempts Made", total_scores)
    
    with stats_col3:
        if total_scores > 0:
            avg_score = sum(st.session_state.user_progress['classification_scores']) / total_scores
            st.metric("Average Score", f"{avg_score:.1f}%")
        else:
            st.metric("Average Score", "N/A")
    
    with stats_col4:
        high_scores = sum(1 for score in st.session_state.user_progress['classification_scores'] if score >= 80)
        st.metric("High Scores (80%+)", high_scores)

    st.markdown("---")
    st.markdown("Created by Mark Connor")

if __name__ == "__main__":
    main()

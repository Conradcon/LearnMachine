import streamlit as st
import pandas as pd
import json
import base64
from typing import Dict, List
import os

class ExerciseCreator:
    def __init__(self):
        self.exercise_file = 'custom_exercises.json'
        self.load_exercises()
    
    def load_exercises(self):
        """Load custom exercises from file"""
        try:
            if os.path.exists(self.exercise_file):
                with open(self.exercise_file, 'r') as f:
                    self.custom_exercises = json.load(f)
            else:
                self.custom_exercises = {}
        except:
            self.custom_exercises = {}
    
    def save_exercises(self):
        """Save custom exercises to file"""
        try:
            with open(self.exercise_file, 'w') as f:
                json.dump(self.custom_exercises, f, indent=2)
        except Exception as e:
            st.error(f"Error saving exercises: {str(e)}")
    
    def run(self):
        st.title("📝 Exercise Creator")
        st.markdown("### Create Custom Learning Activities for Students")
        
        tab1, tab2, tab3 = st.tabs(["🔨 Create Exercise", "📚 Manage Exercises", "🎓 Exercise Templates"])
        
        with tab1:
            self.create_exercise_interface()
        
        with tab2:
            self.manage_exercises_interface()
        
        with tab3:
            self.show_templates()
    
    def create_exercise_interface(self):
        """Interface for creating new exercises"""
        st.markdown("### 🔨 Create New Exercise")
        
        exercise_type = st.selectbox(
            "Choose exercise type:",
            ["Classification", "Pattern Recognition", "Dataset Analysis"]
        )
        
        if exercise_type == "Classification":
            self.create_classification_exercise()
        elif exercise_type == "Pattern Recognition":
            self.create_pattern_exercise()
        elif exercise_type == "Dataset Analysis":
            self.create_analysis_exercise()
    
    def create_classification_exercise(self):
        """Create a classification exercise"""
        st.markdown("#### 🎯 Classification Exercise Builder")
        
        # Exercise metadata
        exercise_name = st.text_input("Exercise Name:", placeholder="e.g., Animals vs Objects")
        exercise_description = st.text_area("Exercise Description:", 
                                          placeholder="Describe what students will learn...")
        
        if not exercise_name:
            st.warning("Please provide an exercise name to continue.")
            return
        
        # Category setup
        st.markdown("**Categories:**")
        num_categories = st.slider("Number of categories:", 2, 4, 2)
        
        categories = {}
        for i in range(num_categories):
            col1, col2, col3 = st.columns(3)
            with col1:
                cat_letter = chr(65 + i)  # A, B, C, D
                st.text_input(f"Category {cat_letter} Name:", key=f"cat_name_{i}")
            with col2:
                st.color_picker(f"Category {cat_letter} Color:", key=f"cat_color_{i}")
            with col3:
                st.text_input(f"Category {cat_letter} Icon (emoji):", key=f"cat_icon_{i}", value="📁")
        
        # Example items
        st.markdown("**Labeled Examples:**")
        examples = {}
        
        for i in range(num_categories):
            cat_letter = chr(65 + i)
            st.markdown(f"**Category {cat_letter} Examples:**")
            
            num_examples = st.slider(f"Number of examples for Category {cat_letter}:", 1, 5, 2, key=f"num_ex_{i}")
            
            examples[cat_letter] = []
            for j in range(num_examples):
                col1, col2 = st.columns(2)
                with col1:
                    item_name = st.text_input(f"Example {j+1} Name:", key=f"ex_{i}_{j}_name")
                with col2:
                    item_description = st.text_input(f"Example {j+1} Description:", key=f"ex_{i}_{j}_desc")
                
                if item_name:
                    examples[cat_letter].append({
                        'name': item_name,
                        'description': item_description,
                        'icon': '🔹'  # Default icon
                    })
        
        # Test items
        st.markdown("**Test Items (for students to classify):**")
        num_test_items = st.slider("Number of test items:", 1, 10, 3)
        
        test_items = []
        for i in range(num_test_items):
            col1, col2, col3 = st.columns(3)
            with col1:
                test_name = st.text_input(f"Test Item {i+1} Name:", key=f"test_{i}_name")
            with col2:
                correct_category = st.selectbox(f"Correct Category:", 
                                              [chr(65 + j) for j in range(num_categories)], 
                                              key=f"test_{i}_cat")
            with col3:
                test_desc = st.text_input(f"Description:", key=f"test_{i}_desc")
            
            if test_name:
                test_items.append({
                    'name': test_name,
                    'description': test_desc,
                    'correct': correct_category,
                    'icon': '❓'
                })
        
        # Save exercise
        if st.button("💾 Save Exercise", type="primary"):
            if self.validate_classification_exercise(exercise_name, examples, test_items):
                exercise_data = {
                    'type': 'classification',
                    'title': exercise_name,
                    'description': exercise_description,
                    'categories': {
                        chr(65 + i): {
                            'name': st.session_state.get(f'cat_name_{i}', f'Category {chr(65 + i)}'),
                            'color': st.session_state.get(f'cat_color_{i}', '#666'),
                            'icon': st.session_state.get(f'cat_icon_{i}', '📁')
                        }
                        for i in range(num_categories)
                    },
                    'labeled_examples': examples,
                    'test_items': test_items,
                    'created_date': pd.Timestamp.now().isoformat()
                }
                
                self.custom_exercises[exercise_name] = exercise_data
                self.save_exercises()
                st.success(f"✅ Exercise '{exercise_name}' saved successfully!")
            else:
                st.error("❌ Please fill in all required fields")
    
    def create_pattern_exercise(self):
        """Create a pattern recognition exercise"""
        st.markdown("#### 🔍 Pattern Recognition Exercise Builder")
        st.info("Pattern recognition exercises help students identify underlying rules or sequences.")
        
        exercise_name = st.text_input("Exercise Name:", placeholder="e.g., Number Sequences")
        exercise_description = st.text_area("Exercise Description:", 
                                          placeholder="Describe the pattern students should find...")
        
        if not exercise_name:
            st.warning("Please provide an exercise name to continue.")
            return
        
        # Pattern type
        pattern_type = st.selectbox(
            "Pattern Type:",
            ["Sequence", "Visual Pattern", "Mathematical Rule", "Custom"]
        )
        
        # Pattern examples
        st.markdown("**Pattern Examples:**")
        num_examples = st.slider("Number of examples:", 3, 8, 5)
        
        examples = []
        for i in range(num_examples):
            col1, col2 = st.columns(2)
            with col1:
                example_input = st.text_input(f"Example {i+1} Input:", key=f"pattern_in_{i}")
            with col2:
                example_output = st.text_input(f"Example {i+1} Output:", key=f"pattern_out_{i}")
            
            if example_input and example_output:
                examples.append({
                    'input': example_input,
                    'output': example_output
                })
        
        # Test cases
        st.markdown("**Test Cases:**")
        num_tests = st.slider("Number of test cases:", 2, 6, 3)
        
        test_cases = []
        for i in range(num_tests):
            col1, col2 = st.columns(2)
            with col1:
                test_input = st.text_input(f"Test {i+1} Input:", key=f"pattern_test_in_{i}")
            with col2:
                test_answer = st.text_input(f"Test {i+1} Answer:", key=f"pattern_test_ans_{i}")
            
            if test_input and test_answer:
                test_cases.append({
                    'input': test_input,
                    'answer': test_answer
                })
        
        # Pattern explanation
        pattern_explanation = st.text_area("Pattern Explanation (for educators):",
                                         placeholder="Explain the underlying rule or pattern...")
        
        # Save exercise
        if st.button("💾 Save Pattern Exercise", type="primary"):
            if exercise_name and examples and test_cases:
                exercise_data = {
                    'type': 'pattern_recognition',
                    'title': exercise_name,
                    'description': exercise_description,
                    'pattern_type': pattern_type,
                    'examples': examples,
                    'test_cases': test_cases,
                    'explanation': pattern_explanation,
                    'created_date': pd.Timestamp.now().isoformat()
                }
                
                self.custom_exercises[exercise_name] = exercise_data
                self.save_exercises()
                st.success(f"✅ Pattern exercise '{exercise_name}' saved successfully!")
            else:
                st.error("❌ Please fill in all required fields")
    
    def create_analysis_exercise(self):
        """Create a dataset analysis exercise"""
        st.markdown("#### 📊 Dataset Analysis Exercise Builder")
        st.info("Create exercises that guide students through data exploration and analysis.")
        
        exercise_name = st.text_input("Exercise Name:", placeholder="e.g., Sales Data Exploration")
        exercise_description = st.text_area("Exercise Description:", 
                                          placeholder="Describe what students will analyze...")
        
        if not exercise_name:
            st.warning("Please provide an exercise name to continue.")
            return
        
        # Dataset source
        dataset_source = st.selectbox(
            "Dataset Source:",
            ["Upload CSV", "Built-in Dataset", "URL"]
        )
        
        dataset_info = {}
        
        if dataset_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV file:", type=['csv'])
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    # Store dataset info
                    dataset_info = {
                        'type': 'uploaded',
                        'filename': uploaded_file.name,
                        'columns': df.columns.tolist(),
                        'dtypes': df.dtypes.to_dict(),
                        'sample_data': df.head().to_dict('records')
                    }
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
        
        elif dataset_source == "Built-in Dataset":
            builtin_options = ["Iris", "Tips", "Housing", "Wine"]
            selected_dataset = st.selectbox("Choose dataset:", builtin_options)
            dataset_info = {'type': 'builtin', 'name': selected_dataset}
        
        elif dataset_source == "URL":
            dataset_url = st.text_input("Dataset URL:")
            if dataset_url:
                dataset_info = {'type': 'url', 'url': dataset_url}
        
        # Analysis tasks
        st.markdown("**Analysis Tasks:**")
        st.info("Define specific tasks for students to complete.")
        
        num_tasks = st.slider("Number of tasks:", 1, 6, 3)
        
        tasks = []
        for i in range(num_tasks):
            st.markdown(f"**Task {i+1}:**")
            col1, col2 = st.columns(2)
            
            with col1:
                task_type = st.selectbox(f"Task Type:", 
                                       ["Data Overview", "Visualization", "Statistical Analysis", "Pattern Discovery"],
                                       key=f"task_type_{i}")
                task_question = st.text_input(f"Question/Instruction:", key=f"task_q_{i}")
            
            with col2:
                task_hint = st.text_area(f"Hint (optional):", key=f"task_hint_{i}")
                expected_answer = st.text_input(f"Expected Answer:", key=f"task_ans_{i}")
            
            if task_question:
                tasks.append({
                    'type': task_type,
                    'question': task_question,
                    'hint': task_hint,
                    'expected_answer': expected_answer
                })
        
        # Learning objectives
        learning_objectives = st.text_area("Learning Objectives:",
                                         placeholder="What should students learn from this exercise?")
        
        # Save exercise
        if st.button("💾 Save Analysis Exercise", type="primary"):
            if exercise_name and dataset_info and tasks:
                exercise_data = {
                    'type': 'dataset_analysis',
                    'title': exercise_name,
                    'description': exercise_description,
                    'dataset_info': dataset_info,
                    'tasks': tasks,
                    'learning_objectives': learning_objectives,
                    'created_date': pd.Timestamp.now().isoformat()
                }
                
                self.custom_exercises[exercise_name] = exercise_data
                self.save_exercises()
                st.success(f"✅ Analysis exercise '{exercise_name}' saved successfully!")
            else:
                st.error("❌ Please fill in all required fields")
    
    def manage_exercises_interface(self):
        """Interface for managing existing exercises"""
        st.markdown("### 📚 Manage Exercises")
        
        if not self.custom_exercises:
            st.info("No custom exercises found. Create your first exercise in the 'Create Exercise' tab!")
            return
        
        # Exercise list
        st.markdown("**Your Exercises:**")
        
        for exercise_name, exercise_data in self.custom_exercises.items():
            with st.expander(f"{exercise_data.get('type', 'Unknown').title()}: {exercise_name}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"**Type:** {exercise_data.get('type', 'Unknown')}")
                    st.markdown(f"**Created:** {exercise_data.get('created_date', 'Unknown')}")
                
                with col2:
                    st.markdown(f"**Description:** {exercise_data.get('description', 'No description')}")
                
                with col3:
                    if st.button(f"🗑️ Delete", key=f"delete_{exercise_name}"):
                        del self.custom_exercises[exercise_name]
                        self.save_exercises()
                        st.rerun()
                    
                    if st.button(f"📋 Copy", key=f"copy_{exercise_name}"):
                        st.code(json.dumps(exercise_data, indent=2))
                
                # Show exercise details
                if exercise_data.get('type') == 'classification':
                    st.markdown("**Categories:**")
                    for cat_id, cat_info in exercise_data.get('categories', {}).items():
                        st.text(f"- {cat_id}: {cat_info.get('name', 'Unknown')}")
                    
                    num_examples = sum(len(examples) for examples in exercise_data.get('labeled_examples', {}).values())
                    num_test_items = len(exercise_data.get('test_items', []))
                    st.markdown(f"**Examples:** {num_examples}, **Test Items:** {num_test_items}")
                
                elif exercise_data.get('type') == 'pattern_recognition':
                    num_examples = len(exercise_data.get('examples', []))
                    num_tests = len(exercise_data.get('test_cases', []))
                    st.markdown(f"**Examples:** {num_examples}, **Test Cases:** {num_tests}")
                
                elif exercise_data.get('type') == 'dataset_analysis':
                    dataset_info = exercise_data.get('dataset_info', {})
                    num_tasks = len(exercise_data.get('tasks', []))
                    st.markdown(f"**Dataset:** {dataset_info.get('type', 'Unknown')}, **Tasks:** {num_tasks}")
    
    def show_templates(self):
        """Show exercise templates and examples"""
        st.markdown("### 🎓 Exercise Templates")
        st.info("Use these templates as starting points for your own exercises.")
        
        template_type = st.selectbox(
            "Choose template category:",
            ["Classification Templates", "Pattern Recognition Templates", "Analysis Templates"]
        )
        
        if template_type == "Classification Templates":
            self.show_classification_templates()
        elif template_type == "Pattern Recognition Templates":
            self.show_pattern_templates()
        elif template_type == "Analysis Templates":
            self.show_analysis_templates()
    
    def show_classification_templates(self):
        """Show classification exercise templates"""
        templates = {
            "Animals vs Objects": {
                "description": "Basic classification between living and non-living things",
                "categories": ["Animals", "Objects"],
                "examples": ["Cat, Dog, Bird", "Car, Phone, Book"],
                "learning_goal": "Understand basic categorization"
            },
            "Geometric Shapes": {
                "description": "Classify geometric shapes by properties",
                "categories": ["Circles", "Polygons", "Irregular"],
                "examples": ["Circle, Oval, Sphere", "Triangle, Square, Pentagon", "Cloud, Blob, Star"],
                "learning_goal": "Recognize geometric properties"
            },
            "Text Sentiment": {
                "description": "Classify text as positive, negative, or neutral",
                "categories": ["Positive", "Negative", "Neutral"],
                "examples": ["Great job!", "This is terrible", "The sky is blue"],
                "learning_goal": "Understand sentiment analysis"
            }
        }
        
        for template_name, template_info in templates.items():
            with st.expander(f"📋 {template_name}"):
                st.markdown(f"**Description:** {template_info['description']}")
                st.markdown(f"**Categories:** {', '.join(template_info['categories'])}")
                st.markdown(f"**Example Items:** {', '.join(template_info['examples'])}")
                st.markdown(f"**Learning Goal:** {template_info['learning_goal']}")
                
                if st.button(f"Use Template", key=f"use_{template_name}"):
                    st.info(f"Template '{template_name}' ready to use! Go to the Create Exercise tab to build your exercise.")
    
    def show_pattern_templates(self):
        """Show pattern recognition templates"""
        templates = {
            "Number Sequences": {
                "description": "Identify patterns in number sequences",
                "examples": "2, 4, 6, 8, ? → 10",
                "pattern_type": "Mathematical progression",
                "learning_goal": "Recognize arithmetic patterns"
            },
            "Shape Sequences": {
                "description": "Visual pattern recognition with shapes",
                "examples": "Circle, Square, Circle, Square, ? → Circle",
                "pattern_type": "Alternating sequence",
                "learning_goal": "Identify visual patterns"
            },
            "Word Patterns": {
                "description": "Find patterns in word relationships",
                "examples": "Cat → Animal, Rose → ?  → Flower",
                "pattern_type": "Category relationships",
                "learning_goal": "Understand semantic relationships"
            }
        }
        
        for template_name, template_info in templates.items():
            with st.expander(f"🔍 {template_name}"):
                st.markdown(f"**Description:** {template_info['description']}")
                st.markdown(f"**Example:** {template_info['examples']}")
                st.markdown(f"**Pattern Type:** {template_info['pattern_type']}")
                st.markdown(f"**Learning Goal:** {template_info['learning_goal']}")
    
    def show_analysis_templates(self):
        """Show dataset analysis templates"""
        templates = {
            "Sales Analysis": {
                "description": "Analyze sales data trends and patterns",
                "tasks": ["Find top-selling products", "Identify seasonal trends", "Calculate growth rates"],
                "skills": "Data exploration, trend analysis, basic statistics",
                "dataset": "Sales transaction data"
            },
            "Student Performance": {
                "description": "Analyze academic performance data",
                "tasks": ["Compare subject scores", "Identify struggling students", "Find correlation between study time and grades"],
                "skills": "Statistical analysis, correlation, data visualization",
                "dataset": "Student grades and study habits"
            },
            "Weather Patterns": {
                "description": "Explore weather data and climate patterns",
                "tasks": ["Identify temperature trends", "Analyze precipitation patterns", "Compare seasonal variations"],
                "skills": "Time series analysis, pattern recognition, climate data interpretation",
                "dataset": "Historical weather data"
            }
        }
        
        for template_name, template_info in templates.items():
            with st.expander(f"📊 {template_name}"):
                st.markdown(f"**Description:** {template_info['description']}")
                st.markdown(f"**Sample Tasks:** {', '.join(template_info['tasks'])}")
                st.markdown(f"**Skills Taught:** {template_info['skills']}")
                st.markdown(f"**Dataset Type:** {template_info['dataset']}")
    
    def validate_classification_exercise(self, name, examples, test_items):
        """Validate classification exercise data"""
        if not name or not examples or not test_items:
            return False
        
        # Check that all categories have examples
        for category, items in examples.items():
            if not items:
                return False
        
        # Check that all test items have correct answers
        for item in test_items:
            if not item.get('name') or not item.get('correct'):
                return False
        
        return True

import streamlit as st
import random
import numpy as np
from typing import Dict, List, Tuple
import base64

class ClassificationExercise:
    def __init__(self):
        self.categories = {
            'A': {'name': 'Transportation', 'color': '#FF6B6B', 'icon': '🚗'},
            'B': {'name': 'Technology', 'color': '#4ECDC4', 'icon': '💻'},
            'C': {'name': 'Nature', 'color': '#45B7D1', 'icon': '🌿'}
        }
        
        # Sample exercises with SVG icons
        self.exercise_data = {
            'transportation_vs_tech': {
                'title': 'Transportation vs Technology',
                'description': 'Classify items as Transportation (A) or Technology (B)',
                'labeled_examples': {
                    'A': [
                        {'name': 'Car', 'icon': self.get_car_svg()},
                        {'name': 'Airplane', 'icon': self.get_airplane_svg()},
                        {'name': 'Boat', 'icon': self.get_boat_svg()}
                    ],
                    'B': [
                        {'name': 'Computer', 'icon': self.get_computer_svg()},
                        {'name': 'Phone', 'icon': self.get_phone_svg()},
                        {'name': 'Camera', 'icon': self.get_camera_svg()}
                    ]
                },
                'test_items': [
                    {'name': 'Bus', 'icon': self.get_bus_svg(), 'correct': 'A'},
                    {'name': 'Tablet', 'icon': self.get_tablet_svg(), 'correct': 'B'},
                    {'name': 'Bicycle', 'icon': self.get_bicycle_svg(), 'correct': 'A'},
                    {'name': 'Router', 'icon': self.get_router_svg(), 'correct': 'B'},
                    {'name': 'Train', 'icon': self.get_train_svg(), 'correct': 'A'}
                ]
            }
        }
    
    def get_car_svg(self):
        return """<svg width="80" height="80" viewBox="0 0 100 100" fill="#FF6B6B">
        <rect x="20" y="40" width="60" height="25" rx="5" fill="currentColor"/>
        <circle cx="30" cy="70" r="8" fill="currentColor"/>
        <circle cx="70" cy="70" r="8" fill="currentColor"/>
        <rect x="25" y="30" width="15" height="15" rx="2" fill="currentColor"/>
        <rect x="60" y="30" width="15" height="15" rx="2" fill="currentColor"/>
        </svg>"""
    
    def get_airplane_svg(self):
        return """<svg width="80" height="80" viewBox="0 0 100 100" fill="#FF6B6B">
        <path d="M50 20 L65 45 L85 40 L90 50 L65 55 L50 80 L35 55 L10 50 L15 40 L35 45 Z" fill="currentColor"/>
        </svg>"""
    
    def get_boat_svg(self):
        return """<svg width="80" height="80" viewBox="0 0 100 100" fill="#FF6B6B">
        <path d="M20 60 L80 60 L75 75 L25 75 Z" fill="currentColor"/>
        <line x1="50" y1="30" x2="50" y2="60" stroke="currentColor" stroke-width="3"/>
        <path d="M50 30 L70 45 L50 40 Z" fill="currentColor"/>
        </svg>"""
    
    def get_computer_svg(self):
        return """<svg width="80" height="80" viewBox="0 0 100 100" fill="#4ECDC4">
        <rect x="15" y="25" width="70" height="45" rx="3" fill="currentColor"/>
        <rect x="20" y="30" width="60" height="35" rx="2" fill="white"/>
        <rect x="40" y="70" width="20" height="8" fill="currentColor"/>
        <rect x="30" y="78" width="40" height="4" fill="currentColor"/>
        </svg>"""
    
    def get_phone_svg(self):
        return """<svg width="80" height="80" viewBox="0 0 100 100" fill="#4ECDC4">
        <rect x="30" y="15" width="40" height="70" rx="8" fill="currentColor"/>
        <rect x="35" y="25" width="30" height="50" rx="3" fill="white"/>
        <circle cx="50" cy="80" r="3" fill="white"/>
        </svg>"""
    
    def get_camera_svg(self):
        return """<svg width="80" height="80" viewBox="0 0 100 100" fill="#4ECDC4">
        <rect x="20" y="30" width="60" height="40" rx="5" fill="currentColor"/>
        <circle cx="50" cy="50" r="12" fill="white"/>
        <circle cx="50" cy="50" r="8" fill="currentColor"/>
        <rect x="40" y="20" width="20" height="10" rx="3" fill="currentColor"/>
        </svg>"""
    
    def get_bus_svg(self):
        return """<svg width="80" height="80" viewBox="0 0 100 100" fill="#666">
        <rect x="15" y="30" width="70" height="30" rx="5" fill="currentColor"/>
        <circle cx="25" cy="65" r="6" fill="currentColor"/>
        <circle cx="75" cy="65" r="6" fill="currentColor"/>
        <rect x="20" y="35" width="12" height="8" rx="1" fill="white"/>
        <rect x="34" y="35" width="12" height="8" rx="1" fill="white"/>
        <rect x="54" y="35" width="12" height="8" rx="1" fill="white"/>
        <rect x="68" y="35" width="12" height="8" rx="1" fill="white"/>
        </svg>"""
    
    def get_tablet_svg(self):
        return """<svg width="80" height="80" viewBox="0 0 100 100" fill="#666">
        <rect x="25" y="15" width="50" height="70" rx="6" fill="currentColor"/>
        <rect x="30" y="25" width="40" height="50" rx="3" fill="white"/>
        <circle cx="50" cy="80" r="2" fill="white"/>
        </svg>"""
    
    def get_bicycle_svg(self):
        return """<svg width="80" height="80" viewBox="0 0 100 100" fill="#666">
        <circle cx="25" cy="65" r="12" fill="none" stroke="currentColor" stroke-width="3"/>
        <circle cx="75" cy="65" r="12" fill="none" stroke="currentColor" stroke-width="3"/>
        <path d="M37 65 L50 45 L50 35 L55 35" stroke="currentColor" stroke-width="3" fill="none"/>
        <path d="M50 45 L63 65" stroke="currentColor" stroke-width="3"/>
        <path d="M45 50 L55 50" stroke="currentColor" stroke-width="2"/>
        </svg>"""
    
    def get_router_svg(self):
        return """<svg width="80" height="80" viewBox="0 0 100 100" fill="#666">
        <rect x="20" y="45" width="60" height="25" rx="3" fill="currentColor"/>
        <circle cx="30" cy="57" r="2" fill="white"/>
        <circle cx="40" cy="57" r="2" fill="white"/>
        <circle cx="70" cy="57" r="2" fill="white"/>
        <line x1="35" y1="30" x2="35" y2="45" stroke="currentColor" stroke-width="2"/>
        <line x1="65" y1="25" x2="65" y2="45" stroke="currentColor" stroke-width="2"/>
        </svg>"""
    
    def get_train_svg(self):
        return """<svg width="80" height="80" viewBox="0 0 100 100" fill="#666">
        <rect x="15" y="35" width="70" height="25" rx="5" fill="currentColor"/>
        <circle cx="25" cy="65" r="5" fill="currentColor"/>
        <circle cx="75" cy="65" r="5" fill="currentColor"/>
        <rect x="20" y="40" width="15" height="10" rx="2" fill="white"/>
        <rect x="40" y="40" width="15" height="10" rx="2" fill="white"/>
        <rect x="60" y="40" width="15" height="10" rx="2" fill="white"/>
        <rect x="25" y="25" width="8" height="12" fill="currentColor"/>
        </svg>"""

    def run(self):
        st.title("🎯 Classification Exercises")
        st.markdown("### Learn Supervised Machine Learning Through Pattern Recognition")
        
        # Exercise selection
        exercise_name = st.selectbox(
            "Choose an exercise:",
            list(self.exercise_data.keys()),
            format_func=lambda x: self.exercise_data[x]['title']
        )
        
        exercise = self.exercise_data[exercise_name]
        
        # Exercise instructions
        with st.expander("📋 Instructions", expanded=True):
            st.markdown(f"""
            **{exercise['title']}**
            
            {exercise['description']}
            
            **How to play:**
            1. Study the labeled examples below to understand the pattern
            2. Use the dropdown menus to classify the unlabeled items
            3. Click 'Submit' to check your answers
            4. Get immediate feedback on your performance
            """)
        
        # Show labeled examples
        st.markdown("### 📚 Labeled Examples - Study These!")
        
        example_cols = st.columns(len(exercise['labeled_examples']))
        
        for idx, (category, examples) in enumerate(exercise['labeled_examples'].items()):
            with example_cols[idx]:
                category_info = self.categories.get(category, {'name': f'Category {category}', 'color': '#666', 'icon': '📁'})
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; border: 3px solid {category_info['color']}; border-radius: 10px; margin: 5px;">
                    <h4>Category {category}: {category_info['name']} {category_info['icon']}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                for example in examples:
                    st.markdown(f"""
                    <div style="text-align: center; margin: 10px; padding: 15px; background-color: #f8f9fa; border-radius: 8px;">
                        {example['icon']}
                        <p style="margin-top: 8px; font-weight: bold;">{example['name']}</p>
                        <span style="background-color: {category_info['color']}; color: white; padding: 3px 8px; border-radius: 12px; font-size: 12px;">Category {category}</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Classification exercise
        st.markdown("### 🔍 Now Classify These Items!")
        st.markdown("Use what you learned from the examples above to classify the following items:")
        
        # Initialize session state for answers
        if f'answers_{exercise_name}' not in st.session_state:
            st.session_state[f'answers_{exercise_name}'] = {}
        
        # Create classification interface
        test_cols = st.columns(min(3, len(exercise['test_items'])))
        
        for idx, item in enumerate(exercise['test_items']):
            col_idx = idx % len(test_cols)
            with test_cols[col_idx]:
                st.markdown(f"""
                <div style="text-align: center; margin: 10px; padding: 15px; background-color: white; border: 2px solid #e0e0e0; border-radius: 8px;">
                    {item['icon']}
                    <p style="margin-top: 8px; font-weight: bold;">{item['name']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Dropdown for classification
                categories = list(exercise['labeled_examples'].keys())
                category_options = [f"Category {cat}: {self.categories.get(cat, {'name': f'Category {cat}'})['name']}" for cat in categories]
                
                selected = st.selectbox(
                    f"Classify '{item['name']}':",
                    ['Select Category'] + category_options,
                    key=f"classify_{exercise_name}_{idx}"
                )
                
                if selected != 'Select Category':
                    category = selected.split(':')[0].replace('Category ', '')
                    st.session_state[f'answers_{exercise_name}'][item['name']] = category
        
        # Submit and feedback
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🔄 Reset Answers", use_container_width=True):
                st.session_state[f'answers_{exercise_name}'] = {}
                st.rerun()
        
        with col2:
            submit_clicked = st.button("✅ Submit Answers", use_container_width=True, type="primary")
        
        with col3:
            if st.button("💡 Show Hints", use_container_width=True):
                st.info("Think about the common characteristics of items in each category from the examples!")
        
        # Process submission
        if submit_clicked:
            self.process_submission(exercise, exercise_name)
        
        # Show previous results if available
        if f'results_{exercise_name}' in st.session_state:
            self.show_results(st.session_state[f'results_{exercise_name}'])

    def process_submission(self, exercise, exercise_name):
        answers = st.session_state.get(f'answers_{exercise_name}', {})
        
        if len(answers) != len(exercise['test_items']):
            st.warning("⚠️ Please classify all items before submitting!")
            return
        
        # Calculate results
        correct = 0
        results = []
        
        for item in exercise['test_items']:
            user_answer = answers.get(item['name'])
            is_correct = user_answer == item['correct']
            
            if is_correct:
                correct += 1
            
            results.append({
                'item': item['name'],
                'icon': item['icon'],
                'user_answer': user_answer,
                'correct_answer': item['correct'],
                'is_correct': is_correct
            })
        
        score = (correct / len(exercise['test_items'])) * 100
        
        # Store results
        st.session_state[f'results_{exercise_name}'] = {
            'score': score,
            'correct': correct,
            'total': len(exercise['test_items']),
            'details': results
        }
        
        # Update progress
        st.session_state.user_progress['classification_scores'].append(score)
        st.session_state.user_progress['exercises_completed'] += 1
        
        st.success(f"🎉 Exercise completed! Your score: {score:.1f}%")
        st.rerun()

    def show_results(self, results):
        st.markdown("---")
        st.markdown("### 📊 Results")
        
        # Score display
        score = results['score']
        if score >= 90:
            st.success(f"🏆 Excellent! Score: {score:.1f}% ({results['correct']}/{results['total']})")
        elif score >= 70:
            st.success(f"✅ Good job! Score: {score:.1f}% ({results['correct']}/{results['total']})")
        elif score >= 50:
            st.warning(f"⚠️ Not bad, but you can improve! Score: {score:.1f}% ({results['correct']}/{results['total']})")
        else:
            st.error(f"❌ Keep practicing! Score: {score:.1f}% ({results['correct']}/{results['total']})")
        
        # Detailed feedback
        st.markdown("### 📝 Detailed Feedback")
        
        for detail in results['details']:
            col1, col2, col3 = st.columns([1, 2, 2])
            
            with col1:
                st.markdown(f"""
                <div style="text-align: center; padding: 10px;">
                    {detail['icon']}
                    <p style="margin: 5px 0; font-weight: bold;">{detail['item']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                status_color = "#28a745" if detail['is_correct'] else "#dc3545"
                status_icon = "✅" if detail['is_correct'] else "❌"
                st.markdown(f"""
                <div style="padding: 10px;">
                    <p><strong>Your answer:</strong> Category {detail['user_answer']}</p>
                    <p><strong>Correct answer:</strong> Category {detail['correct_answer']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                if detail['is_correct']:
                    st.success(f"{status_icon} Correct!")
                else:
                    st.error(f"{status_icon} Incorrect")
        
        # Learning insights
        st.markdown("### 🧠 Learning Insights")
        if score >= 80:
            st.info("💡 You're getting good at pattern recognition! Try more advanced exercises to continue learning.")
        else:
            st.info("💡 Focus on identifying the key characteristics that distinguish each category. Look for common patterns in the examples!")

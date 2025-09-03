import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
import os

# Add data module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
from sample_datasets import SampleDatasets

class ModelFitting:
    def __init__(self):
        self.datasets = SampleDatasets()
        
        self.classification_models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Support Vector Machine': SVC(random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier()
        }
        
        self.regression_models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Support Vector Regression': SVR(),
            'K-Nearest Neighbors': KNeighborsRegressor()
        }
    
    def run(self):
        st.title("🔧 Model Fitting Laboratory")
        st.markdown("### Train and Evaluate Machine Learning Models")
        
        # Show processed dataset status
        if 'processed_df' in st.session_state:
            st.success("✅ Processed dataset available from Dataset Analysis Lab")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Samples", len(st.session_state.processed_df))
            with col2:
                st.metric("Features", len(st.session_state.processed_df.columns))
            with col3:
                numeric_count = len(st.session_state.processed_df.select_dtypes(include=[np.number]).columns)
                st.metric("Numeric Features", numeric_count)
        else:
            st.info("💡 Tip: Process datasets in the Dataset Analysis Lab to use encoded features here!")
        
        # Model type selection
        model_type = st.selectbox(
            "Choose the type of machine learning problem:",
            ["Classification", "Regression"]
        )
        
        if model_type == "Classification":
            self.run_classification()
        else:
            self.run_regression()
    
    def run_classification(self):
        """Run classification model fitting interface"""
        st.markdown("### 🎯 Classification Models")
        st.info("Classification predicts categories or classes. Examples: spam detection, image recognition, medical diagnosis.")
        
        # Dataset selection for classification
        dataset_options = {
            'iris': 'Iris Flower Species',
            'wine': 'Wine Type Classification',
            'breast_cancer': 'Breast Cancer Diagnosis',
            'processed': 'Use Processed Dataset from Analysis Lab',
            'custom': 'Upload Custom Dataset'
        }
        
        selected_dataset = st.selectbox(
            "Choose a classification dataset:",
            list(dataset_options.keys()),
            format_func=lambda x: dataset_options[x]
        )
        
        df, target_col = self.load_classification_dataset(selected_dataset)
        
        if df is not None:
            # Show dataset info
            st.markdown("#### 📊 Dataset Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Samples", len(df))
            with col2:
                st.metric("Features", len(df.columns) - 1)
            with col3:
                st.metric("Classes", df[target_col].nunique())
            
            # Feature selection
            feature_cols = [col for col in df.columns if col != target_col and df[col].dtype in ['int64', 'float64']]
            selected_features = st.multiselect(
                "Select features for training:",
                feature_cols,
                default=feature_cols[:4] if len(feature_cols) > 4 else feature_cols
            )
            
            if len(selected_features) >= 1:
                # Model selection
                selected_model = st.selectbox(
                    "Choose a classification model:",
                    list(self.classification_models.keys())
                )
                
                # Train/test split
                test_size = st.slider("Test set size:", 0.1, 0.5, 0.2, 0.05)
                
                if st.button("🚀 Train Model", type="primary"):
                    self.train_classification_model(
                        df, target_col, selected_features, 
                        selected_model, test_size
                    )
    
    def run_regression(self):
        """Run regression model fitting interface"""
        st.markdown("### 📈 Regression Models")
        st.info("Regression predicts continuous values. Examples: house prices, stock prices, temperature forecasting.")
        
        # Dataset selection for regression
        dataset_options = {
            'housing': 'House Price Prediction',
            'tips': 'Restaurant Tip Amount',
            'processed': 'Use Processed Dataset from Analysis Lab',
            'custom': 'Upload Custom Dataset'
        }
        
        selected_dataset = st.selectbox(
            "Choose a regression dataset:",
            list(dataset_options.keys()),
            format_func=lambda x: dataset_options[x]
        )
        
        df, target_col = self.load_regression_dataset(selected_dataset)
        
        if df is not None:
            # Show dataset info
            st.markdown("#### 📊 Dataset Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Samples", len(df))
            with col2:
                st.metric("Features", len(df.columns) - 1)
            with col3:
                target_range = df[target_col].max() - df[target_col].min()
                st.metric("Target Range", f"{target_range:.2f}")
            
            # Feature selection
            feature_cols = [col for col in df.columns if col != target_col and df[col].dtype in ['int64', 'float64']]
            selected_features = st.multiselect(
                "Select features for training:",
                feature_cols,
                default=feature_cols[:4] if len(feature_cols) > 4 else feature_cols
            )
            
            if len(selected_features) >= 1:
                # Model selection
                selected_model = st.selectbox(
                    "Choose a regression model:",
                    list(self.regression_models.keys())
                )
                
                # Train/test split
                test_size = st.slider("Test set size:", 0.1, 0.5, 0.2, 0.05)
                
                if st.button("🚀 Train Model", type="primary"):
                    self.train_regression_model(
                        df, target_col, selected_features, 
                        selected_model, test_size
                    )
    
    def load_classification_dataset(self, dataset_name):
        """Load classification dataset"""
        if dataset_name == 'iris':
            from sklearn.datasets import load_iris
            data = load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['species'] = pd.Categorical.from_codes(data.target, data.target_names)
            return df, 'species'
        
        elif dataset_name == 'wine':
            from sklearn.datasets import load_wine
            data = load_wine()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['wine_class'] = pd.Categorical.from_codes(data.target, data.target_names)
            return df, 'wine_class'
        
        elif dataset_name == 'breast_cancer':
            from sklearn.datasets import load_breast_cancer
            data = load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['diagnosis'] = pd.Categorical.from_codes(data.target, data.target_names)
            return df, 'diagnosis'
        
        elif dataset_name == 'processed':
            # Use processed dataset from Dataset Analysis Lab
            if 'processed_df' in st.session_state:
                df = st.session_state.processed_df.copy()
                
                # Let user select target column from the processed dataset
                if len(df.columns) > 1:
                    target_column = st.selectbox(
                        "Select the target/label column:",
                        df.columns,
                        help="Choose which column contains the classes you want to predict"
                    )
                    return df, target_column
                else:
                    st.warning("Processed dataset must have at least 2 columns (features and target).")
                    return None, None
            else:
                st.warning("No processed dataset found. Please visit the Dataset Analysis Lab first and process a dataset.")
                st.info("💡 Go to Dataset Analysis → Data Preprocessing → Apply encoding transformations")
                return None, None
        
        elif dataset_name == 'custom':
            uploaded_file = st.file_uploader(
                "Upload your CSV dataset for classification:",
                type=['csv'],
                help="Upload a CSV file with your classification data. The first row should contain column headers."
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    # Let user select target column
                    if len(df.columns) > 1:
                        target_column = st.selectbox(
                            "Select the target/label column:",
                            df.columns,
                            help="Choose which column contains the classes you want to predict"
                        )
                        return df, target_column
                    else:
                        st.warning("Dataset must have at least 2 columns (features and target).")
                        return None, None
                        
                except Exception as e:
                    st.error(f"Error reading the uploaded file: {str(e)}")
                    st.info("Please make sure your file is a valid CSV with proper headers.")
                    return None, None
            
            return None, None
        
        return None, None
    
    def load_regression_dataset(self, dataset_name):
        """Load regression dataset"""
        if dataset_name == 'housing':
            df = self.datasets.get_housing_dataset()
            return df, 'price'
        
        elif dataset_name == 'tips':
            df = self.datasets.get_tips_dataset()
            return df, 'tip'
        
        elif dataset_name == 'processed':
            # Use processed dataset from Dataset Analysis Lab
            if 'processed_df' in st.session_state:
                df = st.session_state.processed_df.copy()
                
                # Let user select target column from the processed dataset
                if len(df.columns) > 1:
                    # Filter to numeric columns for regression target
                    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                    if len(numeric_cols) > 0:
                        target_column = st.selectbox(
                            "Select the target/label column:",
                            numeric_cols,
                            help="Choose which numeric column contains the values you want to predict"
                        )
                        return df, target_column
                    else:
                        st.warning("No numeric columns found in processed dataset for regression target.")
                        return None, None
                else:
                    st.warning("Processed dataset must have at least 2 columns (features and target).")
                    return None, None
            else:
                st.warning("No processed dataset found. Please visit the Dataset Analysis Lab first and process a dataset.")
                st.info("💡 Go to Dataset Analysis → Data Preprocessing → Apply encoding transformations")
                return None, None
        
        elif dataset_name == 'custom':
            uploaded_file = st.file_uploader(
                "Upload your CSV dataset for regression:",
                type=['csv'],
                help="Upload a CSV file with your regression data. The first row should contain column headers."
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    # Let user select target column
                    if len(df.columns) > 1:
                        # Filter to numeric columns for regression target
                        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                        if len(numeric_cols) > 0:
                            target_column = st.selectbox(
                                "Select the target/label column:",
                                numeric_cols,
                                help="Choose which numeric column contains the values you want to predict"
                            )
                            return df, target_column
                        else:
                            st.warning("No numeric columns found for regression target.")
                            return None, None
                    else:
                        st.warning("Dataset must have at least 2 columns (features and target).")
                        return None, None
                        
                except Exception as e:
                    st.error(f"Error reading the uploaded file: {str(e)}")
                    st.info("Please make sure your file is a valid CSV with proper headers.")
                    return None, None
            
            return None, None
        
        return None, None
    
    def train_classification_model(self, df, target_col, features, model_name, test_size):
        """Train and evaluate classification model"""
        try:
            # Prepare data
            X = df[features]
            y = df[target_col]
            
            # Encode target if categorical
            if y.dtype == 'object' or y.dtype.name == 'category':
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                class_names = le.classes_
            else:
                y_encoded = y
                class_names = None
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = self.classification_models[model_name]
            
            with st.spinner(f"Training {model_name}..."):
                model.fit(X_train_scaled, y_train)
                
                # Predictions
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)
                
                # Probabilities (if available)
                train_proba = None
                test_proba = None
                if hasattr(model, 'predict_proba'):
                    train_proba = model.predict_proba(X_train_scaled)
                    test_proba = model.predict_proba(X_test_scaled)
            
            # Display results
            st.success("✅ Model training completed!")
            
            # Performance metrics
            st.markdown("### 📊 Model Performance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                train_accuracy = accuracy_score(y_train, train_pred)
                st.metric("Training Accuracy", f"{train_accuracy:.3f}")
            
            with col2:
                test_accuracy = accuracy_score(y_test, test_pred)
                st.metric("Test Accuracy", f"{test_accuracy:.3f}")
            
            # Overfitting check
            if train_accuracy - test_accuracy > 0.1:
                st.warning("⚠️ Model might be overfitting (large gap between training and test accuracy)")
            elif test_accuracy > train_accuracy:
                st.info("ℹ️ Test accuracy higher than training - this can happen with small datasets")
            else:
                st.success("✅ Model shows good generalization")
            
            # Confusion Matrix
            st.markdown("### 🎯 Confusion Matrix")
            cm = confusion_matrix(y_test, test_pred)
            
            if class_names is not None:
                labels = class_names
            else:
                labels = [f"Class {i}" for i in range(len(np.unique(y_test)))]
            
            fig = px.imshow(
                cm,
                x=labels,
                y=labels,
                color_continuous_scale='Blues',
                title="Confusion Matrix (Test Set)"
            )
            fig.update_layout(
                xaxis_title="Predicted",
                yaxis_title="Actual"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Classification Report
            st.markdown("### 📋 Detailed Classification Report")
            if class_names is not None:
                report = classification_report(y_test, test_pred, target_names=labels, output_dict=True)
            else:
                report = classification_report(y_test, test_pred, output_dict=True)
            
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(3), use_container_width=True)
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                st.markdown("### 🔍 Feature Importance")
                
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Decision boundary visualization (for 2D data)
            if len(features) == 2:
                st.markdown("### 🎨 Decision Boundary Visualization")
                self.plot_decision_boundary(X_test, y_test, model, scaler, features, class_names)
            
            # Model interpretation
            st.markdown("### 🧠 Model Interpretation")
            self.explain_classification_model(model_name, test_accuracy, len(features))
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
    
    def train_regression_model(self, df, target_col, features, model_name, test_size):
        """Train and evaluate regression model"""
        try:
            # Prepare data
            X = df[features]
            y = df[target_col]
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = self.regression_models[model_name]
            
            with st.spinner(f"Training {model_name}..."):
                model.fit(X_train_scaled, y_train)
                
                # Predictions
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)
            
            # Display results
            st.success("✅ Model training completed!")
            
            # Performance metrics
            st.markdown("### 📊 Model Performance")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                train_r2 = r2_score(y_train, train_pred)
                st.metric("Training R²", f"{train_r2:.3f}")
            
            with col2:
                test_r2 = r2_score(y_test, test_pred)
                st.metric("Test R²", f"{test_r2:.3f}")
            
            with col3:
                test_mae = mean_absolute_error(y_test, test_pred)
                st.metric("Test MAE", f"{test_mae:.3f}")
            
            # Additional metrics
            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)
            test_rmse = np.sqrt(test_mse)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Training MSE", f"{train_mse:.3f}")
            with col2:
                st.metric("Test MSE", f"{test_mse:.3f}")
            with col3:
                st.metric("Test RMSE", f"{test_rmse:.3f}")
            
            # Overfitting check
            if train_r2 - test_r2 > 0.2:
                st.warning("⚠️ Model might be overfitting (large gap between training and test R²)")
            else:
                st.success("✅ Model shows good generalization")
            
            # Prediction vs Actual plot
            st.markdown("### 🎯 Predictions vs Actual Values")
            
            pred_df = pd.DataFrame({
                'Actual': y_test,
                'Predicted': test_pred,
                'Set': 'Test'
            })
            
            # Add training data for comparison
            train_df = pd.DataFrame({
                'Actual': y_train,
                'Predicted': train_pred,
                'Set': 'Train'
            })
            
            combined_df = pd.concat([pred_df, train_df])
            
            fig = px.scatter(
                combined_df,
                x='Actual',
                y='Predicted',
                color='Set',
                title='Predicted vs Actual Values'
            )
            
            # Add perfect prediction line
            min_val = min(combined_df['Actual'].min(), combined_df['Predicted'].min())
            max_val = max(combined_df['Actual'].max(), combined_df['Predicted'].max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash', color='red')
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Residuals plot
            st.markdown("### 📊 Residual Analysis")
            
            residuals = y_test - test_pred
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Residuals vs Predicted
                fig = px.scatter(
                    x=test_pred,
                    y=residuals,
                    title='Residuals vs Predicted Values'
                )
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                fig.update_layout(
                    xaxis_title='Predicted Values',
                    yaxis_title='Residuals'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Residuals histogram
                fig = px.histogram(
                    x=residuals,
                    title='Residuals Distribution',
                    marginal='box'
                )
                fig.update_layout(xaxis_title='Residuals')
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                st.markdown("### 🔍 Feature Importance")
                
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Coefficients for linear regression
            elif hasattr(model, 'coef_'):
                st.markdown("### 🔍 Model Coefficients")
                
                coef_df = pd.DataFrame({
                    'Feature': features,
                    'Coefficient': model.coef_
                }).sort_values('Coefficient', ascending=False, key=abs)
                
                fig = px.bar(
                    coef_df,
                    x='Coefficient',
                    y='Feature',
                    orientation='h',
                    title='Feature Coefficients'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Model interpretation
            st.markdown("### 🧠 Model Interpretation")
            self.explain_regression_model(model_name, test_r2, test_rmse, len(features))
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
    
    def plot_decision_boundary(self, X, y, model, scaler, features, class_names):
        """Plot decision boundary for 2D classification"""
        try:
            # Create a mesh
            h = 0.02
            x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
            y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                               np.arange(y_min, y_max, h))
            
            # Make predictions on the mesh
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            mesh_points_scaled = scaler.transform(mesh_points)
            Z = model.predict(mesh_points_scaled)
            Z = Z.reshape(xx.shape)
            
            # Create the plot
            fig = go.Figure()
            
            # Add contour plot for decision boundary
            fig.add_trace(go.Contour(
                x=np.arange(x_min, x_max, h),
                y=np.arange(y_min, y_max, h),
                z=Z,
                showscale=False,
                opacity=0.3,
                hoverinfo='skip'
            ))
            
            # Add scatter plot for data points
            if class_names is not None:
                labels = [class_names[int(label)] for label in y]
            else:
                labels = [f"Class {int(label)}" for label in y]
            
            fig.add_trace(go.Scatter(
                x=X.iloc[:, 0],
                y=X.iloc[:, 1],
                mode='markers',
                marker=dict(
                    color=y,
                    colorscale='viridis',
                    size=8,
                    line=dict(width=1, color='white')
                ),
                text=labels,
                hovertemplate='<b>%{text}</b><br>' +
                            f'{features[0]}: %{{x}}<br>' +
                            f'{features[1]}: %{{y}}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Decision Boundary Visualization',
                xaxis_title=features[0],
                yaxis_title=features[1],
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating decision boundary plot: {str(e)}")
    
    def explain_classification_model(self, model_name, accuracy, num_features):
        """Provide educational explanation of classification model"""
        explanations = {
            'Logistic Regression': f"""
            **Logistic Regression** uses a mathematical function to model the probability of class membership.
            
            - **Strengths**: Fast, interpretable, works well with linearly separable data
            - **Your Results**: {accuracy:.1%} accuracy with {num_features} features
            - **When to use**: When you need interpretable results and have linear relationships
            """,
            
            'Random Forest': f"""
            **Random Forest** combines many decision trees to make more robust predictions.
            
            - **Strengths**: Handles non-linear relationships, provides feature importance, resistant to overfitting
            - **Your Results**: {accuracy:.1%} accuracy with {num_features} features
            - **When to use**: When you have complex data with non-linear patterns
            """,
            
            'Support Vector Machine': f"""
            **Support Vector Machine (SVM)** finds the optimal boundary between classes.
            
            - **Strengths**: Effective in high dimensions, memory efficient, versatile with different kernels
            - **Your Results**: {accuracy:.1%} accuracy with {num_features} features
            - **When to use**: When you have high-dimensional data or need robust classification
            """,
            
            'K-Nearest Neighbors': f"""
            **K-Nearest Neighbors (KNN)** classifies based on the majority class of nearest neighbors.
            
            - **Strengths**: Simple concept, no assumptions about data distribution, works well locally
            - **Your Results**: {accuracy:.1%} accuracy with {num_features} features
            - **When to use**: When local patterns are important and you have sufficient data
            """
        }
        
        st.info(explanations.get(model_name, "Model explanation not available."))
        
        # Performance interpretation
        if accuracy >= 0.9:
            st.success("🎉 Excellent performance! Your model is making very accurate predictions.")
        elif accuracy >= 0.8:
            st.success("✅ Good performance! Your model is working well.")
        elif accuracy >= 0.7:
            st.warning("⚠️ Moderate performance. Consider feature engineering or trying different models.")
        else:
            st.error("❌ Low performance. The model may need more data, better features, or different algorithms.")
    
    def explain_regression_model(self, model_name, r2_score, rmse, num_features):
        """Provide educational explanation of regression model"""
        explanations = {
            'Linear Regression': f"""
            **Linear Regression** models the relationship between features and target using a straight line.
            
            - **Strengths**: Fast, interpretable, good baseline model
            - **Your Results**: R² = {r2_score:.3f}, RMSE = {rmse:.3f} with {num_features} features
            - **When to use**: When relationships are linear and you need interpretable results
            """,
            
            'Random Forest': f"""
            **Random Forest Regression** combines many decision trees for more accurate predictions.
            
            - **Strengths**: Handles non-linear relationships, provides feature importance, robust to outliers
            - **Your Results**: R² = {r2_score:.3f}, RMSE = {rmse:.3f} with {num_features} features
            - **When to use**: When you have complex non-linear relationships in your data
            """,
            
            'Support Vector Regression': f"""
            **Support Vector Regression (SVR)** finds the best fit within a margin of tolerance.
            
            - **Strengths**: Effective in high dimensions, robust to outliers, memory efficient
            - **Your Results**: R² = {r2_score:.3f}, RMSE = {rmse:.3f} with {num_features} features
            - **When to use**: When you have high-dimensional data or need robust predictions
            """,
            
            'K-Nearest Neighbors': f"""
            **K-Nearest Neighbors Regression** predicts based on the average of nearest neighbors.
            
            - **Strengths**: Simple concept, captures local patterns, no assumptions about data
            - **Your Results**: R² = {r2_score:.3f}, RMSE = {rmse:.3f} with {num_features} features
            - **When to use**: When local patterns are important and you have sufficient data
            """
        }
        
        st.info(explanations.get(model_name, "Model explanation not available."))
        
        # Performance interpretation
        if r2_score >= 0.8:
            st.success("🎉 Excellent fit! Your model explains most of the variance in the data.")
        elif r2_score >= 0.6:
            st.success("✅ Good fit! Your model captures the main patterns.")
        elif r2_score >= 0.4:
            st.warning("⚠️ Moderate fit. Consider feature engineering or trying different models.")
        else:
            st.error("❌ Poor fit. The model may need more relevant features or different algorithms.")

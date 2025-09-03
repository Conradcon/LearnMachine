import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.decomposition import PCA
import sys
import os

# Add data module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
from sample_datasets import SampleDatasets

class DatasetAnalysis:
    def __init__(self):
        self.datasets = SampleDatasets()
        
    def run(self):
        st.title("📊 Dataset Analysis Lab")
        st.markdown("### Explore Real-World Data and Discover Patterns")
        
        # Dataset selection options
        data_source = st.radio(
            "Choose data source:",
            ["Built-in Datasets", "Upload Custom Dataset"],
            horizontal=True
        )
        
        df, target_column, dataset_info = None, None, None
        selected_dataset = None
        
        if data_source == "Built-in Datasets":
            # Built-in dataset selection
            dataset_options = {
                'iris': 'Iris Flower Dataset (Classification)',
                'wine': 'Wine Classification Dataset',
                'breast_cancer': 'Breast Cancer Dataset',
                'tips': 'Restaurant Tips Dataset',
                'housing': 'Boston Housing Prices'
            }
            
            selected_dataset = st.selectbox(
                "Choose a dataset to analyze:",
                list(dataset_options.keys()),
                format_func=lambda x: dataset_options[x]
            )
            
            # Load selected dataset
            df, target_column, dataset_info = self.load_dataset(selected_dataset)
            
        else:
            # Custom dataset upload
            uploaded_file = st.file_uploader(
                "Upload your CSV dataset:",
                type=['csv'],
                help="Upload a CSV file with your data. The first row should contain column headers."
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    # Let user select target column
                    if len(df.columns) > 1:
                        target_column = st.selectbox(
                            "Select the target/label column:",
                            df.columns,
                            help="Choose which column contains the values you want to predict or analyze"
                        )
                        
                        # Determine task type
                        if pd.api.types.is_numeric_dtype(df[target_column]):
                            unique_values = df[target_column].nunique()
                            if unique_values <= 10:
                                task_type = "Classification"
                            else:
                                task_type = "Regression"
                        else:
                            task_type = "Classification"
                        
                        dataset_info = {
                            'name': uploaded_file.name,
                            'description': f'Custom uploaded dataset: {uploaded_file.name}',
                            'task': task_type,
                            'features': len(df.columns) - 1,
                            'samples': len(df),
                            'classes': df[target_column].nunique() if task_type == "Classification" else "N/A"
                        }
                        
                    else:
                        st.warning("Dataset must have at least 2 columns (features and target).")
                        df = None
                        
                except Exception as e:
                    st.error(f"Error reading the uploaded file: {str(e)}")
                    st.info("Please make sure your file is a valid CSV with proper headers.")
                    df = None
        
        if df is not None:
            # Dataset overview
            self.show_dataset_overview(df, target_column, dataset_info)
            
            # Store processed dataframe in session state (reset if new dataset)
            dataset_key = f"{data_source}_{selected_dataset if data_source == 'Built-in Datasets' else 'custom'}"
            if 'current_dataset_key' not in st.session_state or st.session_state.current_dataset_key != dataset_key:
                st.session_state.current_dataset_key = dataset_key
                st.session_state.processed_df = df.copy()
            elif 'processed_df' not in st.session_state:
                st.session_state.processed_df = df.copy()
            
            # Analysis tabs
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📋 Data Preview", 
                "🔧 Data Preprocessing",
                "📈 Visualizations", 
                "🔍 Feature Analysis", 
                "🎯 Target Analysis",
                "🧮 Advanced Analysis"
            ])
            
            with tab1:
                self.show_data_preview(st.session_state.processed_df, target_column)
            
            with tab2:
                st.session_state.processed_df = self.show_data_preprocessing(st.session_state.processed_df, target_column)
            
            with tab3:
                self.show_visualizations(st.session_state.processed_df, target_column)
            
            with tab4:
                self.show_feature_analysis(st.session_state.processed_df, target_column)
            
            with tab5:
                self.show_target_analysis(st.session_state.processed_df, target_column)
            
            with tab6:
                self.show_advanced_analysis(st.session_state.processed_df, target_column)
    
    def load_dataset(self, dataset_name):
        """Load the selected dataset"""
        try:
            if dataset_name == 'iris':
                data = load_iris()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df['species'] = data.target
                df['species_name'] = pd.Categorical.from_codes(data.target, data.target_names)
                return df, 'species_name', {
                    'name': 'Iris Flower Dataset',
                    'description': 'Classic dataset with measurements of iris flowers',
                    'task': 'Multi-class classification',
                    'features': 4,
                    'samples': len(df),
                    'classes': 3
                }
            
            elif dataset_name == 'wine':
                data = load_wine()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df['wine_class'] = data.target
                df['wine_class_name'] = pd.Categorical.from_codes(data.target, data.target_names)
                return df, 'wine_class_name', {
                    'name': 'Wine Classification Dataset',
                    'description': 'Chemical analysis of wines from different regions',
                    'task': 'Multi-class classification',
                    'features': 13,
                    'samples': len(df),
                    'classes': 3
                }
            
            elif dataset_name == 'breast_cancer':
                data = load_breast_cancer()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df['diagnosis'] = data.target
                df['diagnosis_name'] = pd.Categorical.from_codes(data.target, data.target_names)
                return df, 'diagnosis_name', {
                    'name': 'Breast Cancer Dataset',
                    'description': 'Features computed from breast cancer cell images',
                    'task': 'Binary classification',
                    'features': 30,
                    'samples': len(df),
                    'classes': 2
                }
            
            elif dataset_name == 'tips':
                df = self.datasets.get_tips_dataset()
                return df, 'time', {
                    'name': 'Restaurant Tips Dataset',
                    'description': 'Tips received by restaurant servers',
                    'task': 'Regression/Classification',
                    'features': 6,
                    'samples': len(df),
                    'classes': 'Various'
                }
            
            elif dataset_name == 'housing':
                df = self.datasets.get_housing_dataset()
                return df, 'price_category', {
                    'name': 'Housing Prices Dataset',
                    'description': 'Factors affecting house prices',
                    'task': 'Regression',
                    'features': 8,
                    'samples': len(df),
                    'classes': 'Continuous'
                }
            
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            return None, None, None
    
    def show_dataset_overview(self, df, target_column, dataset_info):
        """Show dataset overview and metadata"""
        st.markdown("### 📋 Dataset Overview")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"""
            **Dataset:** {dataset_info['name']}
            
            **Description:** {dataset_info['description']}
            
            **Machine Learning Task:** {dataset_info['task']}
            """)
        
        with col2:
            # Key statistics
            st.markdown("**Dataset Statistics:**")
            st.metric("Samples", dataset_info['samples'])
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                st.metric("Features", dataset_info['features'])
            with col2_2:
                st.metric("Classes", dataset_info['classes'])
    
    def show_data_preview(self, df, target_column):
        """Show data preview and basic information"""
        st.markdown("### 👀 Data Preview")
        
        # Sample rows
        st.markdown("**Sample Data:**")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Basic info
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Data Types:**")
            dtype_df = pd.DataFrame({
                'Column': df.dtypes.index,
                'Data Type': df.dtypes.astype(str).values,
                'Non-Null Count': df.count().values
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        with col2:
            st.markdown("**Missing Values:**")
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing Count': df.isnull().sum().values,
                'Missing %': (df.isnull().sum() / len(df) * 100).round(2).values
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0]
            
            if len(missing_df) > 0:
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("✅ No missing values found!")
        
        # Summary statistics
        st.markdown("**Summary Statistics:**")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    
    def show_data_preprocessing(self, df, target_column):
        """Show data preprocessing options including encoding"""
        st.markdown("### 🔧 Data Preprocessing")
        st.info("Transform non-numeric features into numeric formats for machine learning algorithms.")
        
        # Make a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Find categorical columns (excluding target)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if target_column in categorical_cols:
            categorical_cols.remove(target_column)
        
        if len(categorical_cols) > 0:
            st.markdown("#### 📝 Available Categorical Features:")
            
            # Show current categorical columns info
            cat_info = []
            for col in categorical_cols:
                unique_count = df[col].nunique()
                sample_values = df[col].unique()[:3]
                cat_info.append({
                    'Column': col,
                    'Unique Values': unique_count,
                    'Sample Values': ', '.join(map(str, sample_values)) + ('...' if unique_count > 3 else '')
                })
            
            cat_df = pd.DataFrame(cat_info)
            st.dataframe(cat_df, use_container_width=True)
            
            st.markdown("#### 🔄 Apply Encoding:")
            
            encoding_options = ["None", "Label Encoding", "Ordinal Encoding"]
            
            # Create encoding selections for each categorical column
            encoding_choices = {}
            for col in categorical_cols:
                encoding_choices[col] = st.selectbox(
                    f"Encoding for '{col}':",
                    encoding_options,
                    key=f"encoding_{col}"
                )
            
            # Apply encodings button
            if st.button("🚀 Apply Encoding Transformations", type="primary"):
                with st.spinner("Applying encoding transformations..."):
                    encoding_info = []
                    
                    for col, encoding_type in encoding_choices.items():
                        if encoding_type == "Label Encoding":
                            le = LabelEncoder()
                            processed_df[f"{col}_encoded"] = le.fit_transform(processed_df[col])
                            
                            # Store mapping for reference
                            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                            encoding_info.append({
                                'Column': col,
                                'Encoding': 'Label Encoding',
                                'New Column': f"{col}_encoded",
                                'Mapping': str(mapping)
                            })
                            
                        elif encoding_type == "Ordinal Encoding":
                            # Get unique values for ordinal ordering
                            unique_vals = processed_df[col].unique()
                            
                            # Let user define the order
                            st.markdown(f"**Define order for '{col}' (drag to reorder):**")
                            ordered_values = st.multiselect(
                                f"Order for {col} (select in order from lowest to highest):",
                                unique_vals,
                                default=list(unique_vals),
                                key=f"order_{col}"
                            )
                            
                            if len(ordered_values) == len(unique_vals):
                                oe = OrdinalEncoder(categories=[ordered_values])
                                processed_df[f"{col}_encoded"] = oe.fit_transform(processed_df[[col]]).flatten()
                                
                                mapping = dict(zip(ordered_values, range(len(ordered_values))))
                                encoding_info.append({
                                    'Column': col,
                                    'Encoding': 'Ordinal Encoding',
                                    'New Column': f"{col}_encoded",
                                    'Mapping': str(mapping)
                                })
                    
                    # Show encoding results
                    if encoding_info:
                        st.success("✅ Encoding transformations applied successfully!")
                        
                        st.markdown("#### 📋 Encoding Summary:")
                        encoding_df = pd.DataFrame(encoding_info)
                        st.dataframe(encoding_df, use_container_width=True)
                        
                        # Show before/after comparison
                        st.markdown("#### 🔄 Before/After Comparison:")
                        
                        for info in encoding_info:
                            col = info['Column']
                            new_col = info['New Column']
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**Original '{col}':**")
                                st.write(df[col].value_counts().head())
                            
                            with col2:
                                st.markdown(f"**Encoded '{new_col}':**")
                                st.write(processed_df[new_col].value_counts().head())
                        
                        # Option to remove original categorical columns
                        if st.checkbox("Remove original categorical columns after encoding"):
                            for info in encoding_info:
                                if info['Column'] in processed_df.columns:
                                    processed_df = processed_df.drop(columns=[info['Column']])
                            st.success("Original categorical columns removed.")
        
        else:
            st.info("No categorical features found that need encoding.")
        
        # Show updated dataset info
        st.markdown("#### 📊 Updated Dataset Information:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Columns", len(processed_df.columns))
        with col2:
            numeric_count = len(processed_df.select_dtypes(include=[np.number]).columns)
            st.metric("Numeric Columns", numeric_count)
        with col3:
            categorical_count = len(processed_df.select_dtypes(include=['object', 'category']).columns)
            st.metric("Categorical Columns", categorical_count)
        
        # Show preview of processed data
        st.markdown("#### 👀 Processed Data Preview:")
        st.dataframe(processed_df.head(), use_container_width=True)
        
        return processed_df
    
    def show_visualizations(self, df, target_column):
        """Show various data visualizations"""
        st.markdown("### 📈 Data Visualizations")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target from features if it's numeric
        if target_column in numeric_cols:
            feature_cols = [col for col in numeric_cols if col not in [target_column, target_column.replace('_name', '')]]
        else:
            feature_cols = numeric_cols
        
        viz_type = st.selectbox(
            "Choose visualization type:",
            ["Distribution Plots", "Correlation Matrix", "Pairwise Relationships", "Box Plots", "Scatter Plots"]
        )
        
        if viz_type == "Distribution Plots":
            st.markdown("**Feature Distributions:**")
            
            if len(feature_cols) > 0:
                selected_features = st.multiselect(
                    "Select features to plot:",
                    feature_cols,
                    default=feature_cols[:4]  # Default to first 4 features
                )
                
                if selected_features:
                    cols = st.columns(min(2, len(selected_features)))
                    
                    for idx, feature in enumerate(selected_features):
                        col_idx = idx % len(cols)
                        with cols[col_idx]:
                            fig = px.histogram(
                                df, 
                                x=feature, 
                                color=target_column if target_column in categorical_cols else None,
                                title=f'Distribution of {feature}',
                                marginal="box"
                            )
                            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Correlation Matrix":
            if len(feature_cols) >= 2:
                st.markdown("**Feature Correlation Matrix:**")
                
                # Calculate correlation matrix
                corr_matrix = df[feature_cols].corr()
                
                # Create heatmap
                fig = px.imshow(
                    corr_matrix,
                    title="Feature Correlation Matrix",
                    color_continuous_scale="RdBu",
                    aspect="auto"
                )
                fig.update_layout(width=700, height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show strongest correlations
                st.markdown("**Strongest Correlations:**")
                # Get upper triangle of correlation matrix
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                correlations = upper_tri.unstack().dropna().sort_values(key=abs, ascending=False)
                
                corr_df = pd.DataFrame({
                    'Feature 1': [pair[0] for pair in correlations.index[:10]],
                    'Feature 2': [pair[1] for pair in correlations.index[:10]],
                    'Correlation': correlations.values[:10].round(3)
                })
                st.dataframe(corr_df, use_container_width=True)
        
        elif viz_type == "Pairwise Relationships":
            if len(feature_cols) >= 2:
                st.markdown("**Pairwise Feature Relationships:**")
                
                selected_features = st.multiselect(
                    "Select features for pairplot (max 5 recommended):",
                    feature_cols,
                    default=feature_cols[:3]
                )
                
                if len(selected_features) >= 2:
                    # Create scatter matrix
                    fig = px.scatter_matrix(
                        df,
                        dimensions=selected_features,
                        color=target_column if target_column in categorical_cols else None,
                        title="Pairwise Feature Relationships"
                    )
                    fig.update_layout(width=800, height=800)
                    st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Box Plots":
            if len(feature_cols) > 0 and target_column in categorical_cols:
                st.markdown("**Box Plots by Target Variable:**")
                
                selected_feature = st.selectbox("Select feature:", feature_cols)
                
                fig = px.box(
                    df,
                    x=target_column,
                    y=selected_feature,
                    title=f'{selected_feature} by {target_column}'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Scatter Plots":
            if len(feature_cols) >= 2:
                st.markdown("**Feature Scatter Plots:**")
                
                col1, col2 = st.columns(2)
                with col1:
                    x_feature = st.selectbox("X-axis feature:", feature_cols, index=0)
                with col2:
                    y_feature = st.selectbox("Y-axis feature:", feature_cols, index=1)
                
                fig = px.scatter(
                    df,
                    x=x_feature,
                    y=y_feature,
                    color=target_column if target_column in categorical_cols else None,
                    title=f'{y_feature} vs {x_feature}'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def show_feature_analysis(self, df, target_column):
        """Detailed analysis of individual features"""
        st.markdown("### 🔍 Feature Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target from features if it's numeric
        if target_column in numeric_cols:
            feature_cols = [col for col in numeric_cols if col not in [target_column, target_column.replace('_name', '')]]
        else:
            feature_cols = numeric_cols
        
        if len(feature_cols) > 0:
            selected_feature = st.selectbox("Select feature to analyze:", feature_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Statistics for {selected_feature}:**")
                stats = df[selected_feature].describe()
                stats_df = pd.DataFrame({
                    'Statistic': stats.index,
                    'Value': stats.values.round(4)
                })
                st.dataframe(stats_df, use_container_width=True)
                
                # Feature importance insights
                st.markdown("**Feature Insights:**")
                mean_val = df[selected_feature].mean()
                std_val = df[selected_feature].std()
                skewness = df[selected_feature].skew()
                
                if abs(skewness) > 1:
                    st.info(f"📊 This feature is {'right' if skewness > 0 else 'left'} skewed (skewness: {skewness:.2f})")
                else:
                    st.success(f"📊 This feature is approximately normally distributed (skewness: {skewness:.2f})")
                
                # Outlier detection
                Q1 = df[selected_feature].quantile(0.25)
                Q3 = df[selected_feature].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[selected_feature] < Q1 - 1.5*IQR) | (df[selected_feature] > Q3 + 1.5*IQR)]
                
                if len(outliers) > 0:
                    st.warning(f"⚠️ Found {len(outliers)} potential outliers ({len(outliers)/len(df)*100:.1f}% of data)")
                else:
                    st.success("✅ No significant outliers detected")
            
            with col2:
                # Distribution plot
                fig = px.histogram(
                    df,
                    x=selected_feature,
                    title=f'Distribution of {selected_feature}',
                    marginal="box"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature by target
                if target_column in categorical_cols:
                    fig2 = px.violin(
                        df,
                        x=target_column,
                        y=selected_feature,
                        title=f'{selected_feature} by {target_column}'
                    )
                    st.plotly_chart(fig2, use_container_width=True)
    
    def show_target_analysis(self, df, target_column):
        """Analysis of the target variable"""
        st.markdown("### 🎯 Target Variable Analysis")
        
        if target_column in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Target Variable: {target_column}**")
                
                if df[target_column].dtype in ['object', 'category']:
                    # Categorical target
                    value_counts = df[target_column].value_counts()
                    st.markdown("**Class Distribution:**")
                    
                    counts_df = pd.DataFrame({
                        'Class': value_counts.index,
                        'Count': value_counts.values,
                        'Percentage': (value_counts.values / len(df) * 100).round(2)
                    })
                    st.dataframe(counts_df, use_container_width=True)
                    
                    # Check for class imbalance
                    max_class = value_counts.max()
                    min_class = value_counts.min()
                    imbalance_ratio = max_class / min_class
                    
                    if imbalance_ratio > 2:
                        st.warning(f"⚠️ Class imbalance detected! Ratio: {imbalance_ratio:.1f}:1")
                    else:
                        st.success("✅ Classes are relatively balanced")
                
                else:
                    # Numeric target
                    stats = df[target_column].describe()
                    stats_df = pd.DataFrame({
                        'Statistic': stats.index,
                        'Value': stats.values.round(4)
                    })
                    st.dataframe(stats_df, use_container_width=True)
            
            with col2:
                # Target distribution plot
                if df[target_column].dtype in ['object', 'category']:
                    fig = px.bar(
                        x=df[target_column].value_counts().index,
                        y=df[target_column].value_counts().values,
                        title=f'Distribution of {target_column}'
                    )
                    fig.update_layout(
                        xaxis_title=target_column,
                        yaxis_title='Count'
                    )
                else:
                    fig = px.histogram(
                        df,
                        x=target_column,
                        title=f'Distribution of {target_column}',
                        marginal="box"
                    )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Target column '{target_column}' not found in dataset")
    
    def show_advanced_analysis(self, df, target_column):
        """Advanced analysis techniques"""
        st.markdown("### 🧮 Advanced Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target from features if it's numeric
        if target_column in numeric_cols:
            feature_cols = [col for col in numeric_cols if col not in [target_column, target_column.replace('_name', '')]]
        else:
            feature_cols = numeric_cols
        
        if len(feature_cols) >= 2:
            analysis_type = st.selectbox(
                "Choose advanced analysis:",
                ["Principal Component Analysis (PCA)", "Feature Importance", "Clustering Visualization"]
            )
            
            if analysis_type == "Principal Component Analysis (PCA)":
                st.markdown("**Principal Component Analysis:**")
                st.info("PCA reduces dimensionality while preserving variance. Useful for visualization and understanding feature relationships.")
                
                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df[feature_cols])
                
                # Apply PCA
                pca = PCA()
                X_pca = pca.fit_transform(X_scaled)
                
                # Explained variance
                explained_var_df = pd.DataFrame({
                    'Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
                    'Explained Variance (%)': (pca.explained_variance_ratio_ * 100).round(2),
                    'Cumulative Variance (%)': (pca.explained_variance_ratio_.cumsum() * 100).round(2)
                })
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(explained_var_df.head(10), use_container_width=True)
                
                with col2:
                    # Scree plot
                    fig = px.line(
                        x=explained_var_df['Component'][:10],
                        y=explained_var_df['Explained Variance (%)'][:10],
                        title='Scree Plot - Explained Variance by Component',
                        markers=True
                    )
                    fig.update_layout(
                        xaxis_title='Principal Component',
                        yaxis_title='Explained Variance (%)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # 2D PCA plot
                if len(feature_cols) > 2:
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                    
                    pca_df = pd.DataFrame({
                        'PC1': X_pca[:, 0],
                        'PC2': X_pca[:, 1]
                    })
                    
                    if target_column in categorical_cols:
                        pca_df[target_column] = df[target_column].values
                        
                        fig = px.scatter(
                            pca_df,
                            x='PC1',
                            y='PC2',
                            color=target_column,
                            title='First Two Principal Components'
                        )
                    else:
                        fig = px.scatter(
                            pca_df,
                            x='PC1',
                            y='PC2',
                            title='First Two Principal Components'
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            elif analysis_type == "Feature Importance":
                st.markdown("**Feature Importance Analysis:**")
                st.info("Understanding which features contribute most to predicting the target variable.")
                
                if target_column in df.select_dtypes(include=['object', 'category']).columns:
                    from sklearn.ensemble import RandomForestClassifier
                    from sklearn.preprocessing import LabelEncoder
                    
                    # Encode target if categorical
                    le = LabelEncoder()
                    y_encoded = le.fit_transform(df[target_column])
                    
                    # Train random forest
                    rf = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf.fit(df[feature_cols], y_encoded)
                    
                    # Get feature importance
                    importance_df = pd.DataFrame({
                        'Feature': feature_cols,
                        'Importance': rf.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.dataframe(importance_df, use_container_width=True)
                    
                    with col2:
                        fig = px.bar(
                            importance_df,
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title='Feature Importance (Random Forest)'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.warning("Feature importance analysis requires a categorical target variable.")
            
            elif analysis_type == "Clustering Visualization":
                st.markdown("**Clustering Visualization:**")
                st.info("Discover natural groupings in your data using K-means clustering.")
                
                from sklearn.cluster import KMeans
                
                n_clusters = st.slider("Number of clusters:", 2, 8, 3)
                
                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df[feature_cols])
                
                # Apply K-means
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(X_scaled)
                
                # PCA for visualization
                if len(feature_cols) > 2:
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)
                    
                    cluster_df = pd.DataFrame({
                        'PC1': X_pca[:, 0],
                        'PC2': X_pca[:, 1],
                        'Cluster': clusters.astype(str)
                    })
                    
                    fig = px.scatter(
                        cluster_df,
                        x='PC1',
                        y='PC2',
                        color='Cluster',
                        title=f'K-means Clustering ({n_clusters} clusters)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Cluster statistics
                    st.markdown("**Cluster Statistics:**")
                    cluster_stats = []
                    for i in range(n_clusters):
                        cluster_data = df[feature_cols][clusters == i]
                        cluster_stats.append({
                            'Cluster': i,
                            'Size': len(cluster_data),
                            'Percentage': f"{len(cluster_data)/len(df)*100:.1f}%"
                        })
                    
                    stats_df = pd.DataFrame(cluster_stats)
                    st.dataframe(stats_df, use_container_width=True)
        
        else:
            st.warning("Advanced analysis requires at least 2 numeric features.")

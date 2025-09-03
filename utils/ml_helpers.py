import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

class MLHelpers:
    """Utility functions for machine learning operations"""
    
    @staticmethod
    def prepare_classification_data(df, target_column, feature_columns=None, test_size=0.2):
        """
        Prepare data for classification tasks
        
        Args:
            df: pandas DataFrame
            target_column: name of target column
            feature_columns: list of feature column names (if None, use all numeric columns)
            test_size: proportion of data for testing
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, scaler, label_encoder)
        """
        # Select features
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in feature_columns:
                feature_columns.remove(target_column)
        
        X = df[feature_columns]
        y = df[target_column]
        
        # Encode target if necessary
        label_encoder = None
        if y.dtype == 'object' or y.dtype.name == 'category':
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
        else:
            y_encoded = y
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder
    
    @staticmethod
    def prepare_regression_data(df, target_column, feature_columns=None, test_size=0.2):
        """
        Prepare data for regression tasks
        
        Args:
            df: pandas DataFrame
            target_column: name of target column
            feature_columns: list of feature column names (if None, use all numeric columns)
            test_size: proportion of data for testing
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, scaler)
        """
        # Select features
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in feature_columns:
                feature_columns.remove(target_column)
        
        X = df[feature_columns]
        y = df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    
    @staticmethod
    def evaluate_classification_model(y_true, y_pred, class_names=None):
        """
        Evaluate classification model performance
        
        Args:
            y_true: true labels
            y_pred: predicted labels
            class_names: list of class names
            
        Returns:
            dict: evaluation metrics
        """
        accuracy = accuracy_score(y_true, y_pred)
        
        # Classification report
        if class_names is not None:
            report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        else:
            report = classification_report(y_true, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    @staticmethod
    def evaluate_regression_model(y_true, y_pred):
        """
        Evaluate regression model performance
        
        Args:
            y_true: true values
            y_pred: predicted values
            
        Returns:
            dict: evaluation metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2
        }
    
    @staticmethod
    def plot_confusion_matrix(cm, class_names=None, title="Confusion Matrix"):
        """
        Create an interactive confusion matrix plot
        
        Args:
            cm: confusion matrix array
            class_names: list of class names
            title: plot title
            
        Returns:
            plotly figure
        """
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(cm))]
        
        fig = px.imshow(
            cm,
            x=class_names,
            y=class_names,
            color_continuous_scale='Blues',
            title=title,
            text_auto=True
        )
        
        fig.update_layout(
            xaxis_title="Predicted",
            yaxis_title="Actual"
        )
        
        return fig
    
    @staticmethod
    def plot_feature_importance(feature_names, importances, title="Feature Importance"):
        """
        Create a feature importance plot
        
        Args:
            feature_names: list of feature names
            importances: array of importance values
            title: plot title
            
        Returns:
            plotly figure
        """
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title=title
        )
        
        return fig
    
    @staticmethod
    def plot_regression_results(y_true, y_pred, title="Predicted vs Actual"):
        """
        Create a regression results plot
        
        Args:
            y_true: true values
            y_pred: predicted values
            title: plot title
            
        Returns:
            plotly figure
        """
        fig = px.scatter(
            x=y_true,
            y=y_pred,
            title=title,
            labels={'x': 'Actual Values', 'y': 'Predicted Values'}
        )
        
        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash', color='red')
        ))
        
        return fig
    
    @staticmethod
    def plot_learning_curve(train_scores, test_scores, train_sizes):
        """
        Create a learning curve plot
        
        Args:
            train_scores: training scores
            test_scores: test scores
            train_sizes: training set sizes
            
        Returns:
            plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=train_scores.mean(axis=1),
            mode='lines+markers',
            name='Training Score',
            error_y=dict(array=train_scores.std(axis=1))
        ))
        
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=test_scores.mean(axis=1),
            mode='lines+markers',
            name='Validation Score',
            error_y=dict(array=test_scores.std(axis=1))
        ))
        
        fig.update_layout(
            title='Learning Curve',
            xaxis_title='Training Set Size',
            yaxis_title='Score'
        )
        
        return fig
    
    @staticmethod
    def calculate_classification_metrics(y_true, y_pred):
        """
        Calculate detailed classification metrics
        
        Args:
            y_true: true labels
            y_pred: predicted labels
            
        Returns:
            dict: detailed metrics
        """
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        # Handle multiclass case
        average_method = 'weighted' if len(np.unique(y_true)) > 2 else 'binary'
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average_method, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average_method, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average=average_method, zero_division=0)
        }
        
        return metrics
    
    @staticmethod
    def create_data_summary(df):
        """
        Create a comprehensive data summary
        
        Args:
            df: pandas DataFrame
            
        Returns:
            dict: data summary information
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        summary = {
            'shape': df.shape,
            'numeric_columns': len(numeric_cols),
            'categorical_columns': len(categorical_cols),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        }
        
        # Column-wise summary
        column_summary = {}
        for col in df.columns:
            col_info = {
                'dtype': str(df[col].dtype),
                'missing_count': df[col].isnull().sum(),
                'missing_percentage': (df[col].isnull().sum() / len(df)) * 100,
                'unique_values': df[col].nunique()
            }
            
            if col in numeric_cols:
                col_info.update({
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'outliers': len(df[(df[col] < df[col].quantile(0.25) - 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))) | 
                                    (df[col] > df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25)))])
                })
            
            elif col in categorical_cols:
                col_info.update({
                    'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                    'most_frequent_count': df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0
                })
            
            column_summary[col] = col_info
        
        summary['column_details'] = column_summary
        
        return summary

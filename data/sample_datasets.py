import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import random

class SampleDatasets:
    """Class to provide sample datasets for machine learning exercises"""
    
    def __init__(self):
        self.random_seed = 42
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
    
    def get_tips_dataset(self):
        """
        Generate a sample restaurant tips dataset
        
        Returns:
            pandas.DataFrame: Tips dataset with realistic data
        """
        # Generate realistic tips data
        np.random.seed(42)
        n_samples = 244
        
        # Generate base data
        total_bills = np.random.normal(20, 8, n_samples)
        total_bills = np.clip(total_bills, 3, 50)  # Reasonable bill range
        
        # Tip percentage based on service quality and other factors
        base_tip_rate = 0.16
        tip_rates = np.random.normal(base_tip_rate, 0.05, n_samples)
        tip_rates = np.clip(tip_rates, 0.05, 0.30)
        
        tips = total_bills * tip_rates
        
        # Generate categorical variables
        times = np.random.choice(['Lunch', 'Dinner'], n_samples, p=[0.4, 0.6])
        days = np.random.choice(['Thur', 'Fri', 'Sat', 'Sun'], n_samples, p=[0.2, 0.2, 0.3, 0.3])
        sexes = np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4])
        smokers = np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])
        
        # Party size affects bill and tip
        sizes = np.random.choice([1, 2, 3, 4, 5, 6], n_samples, p=[0.05, 0.5, 0.2, 0.15, 0.07, 0.03])
        
        # Adjust bills based on party size
        total_bills = total_bills * (0.7 + 0.3 * sizes)
        tips = total_bills * tip_rates
        
        # Create DataFrame
        df = pd.DataFrame({
            'total_bill': np.round(total_bills, 2),
            'tip': np.round(tips, 2),
            'sex': sexes,
            'smoker': smokers,
            'day': days,
            'time': times,
            'size': sizes
        })
        
        # Add some derived features for analysis
        df['tip_percentage'] = np.round((df['tip'] / df['total_bill']) * 100, 1)
        df['bill_per_person'] = np.round(df['total_bill'] / df['size'], 2)
        
        return df
    
    def get_housing_dataset(self):
        """
        Generate a sample housing prices dataset
        
        Returns:
            pandas.DataFrame: Housing dataset with realistic features
        """
        np.random.seed(42)
        n_samples = 506
        
        # Generate housing features
        # Rooms (number of rooms)
        rooms = np.random.normal(6.3, 0.7, n_samples)
        rooms = np.clip(rooms, 3, 10)
        
        # Age of house
        age = np.random.uniform(2, 100, n_samples)
        
        # Distance to employment centers
        distance = np.random.exponential(3.5, n_samples)
        distance = np.clip(distance, 1, 15)
        
        # Property tax rate
        tax_rate = np.random.normal(11, 7, n_samples)
        tax_rate = np.clip(tax_rate, 1, 30)
        
        # Pupil-teacher ratio
        pupil_teacher = np.random.normal(18.5, 2.2, n_samples)
        pupil_teacher = np.clip(pupil_teacher, 12, 25)
        
        # Crime rate
        crime_rate = np.random.exponential(3.5, n_samples)
        crime_rate = np.clip(crime_rate, 0.1, 90)
        
        # Percentage of lower status population
        lower_status = np.random.beta(2, 5, n_samples) * 40
        
        # Air quality (nitric oxides concentration)
        air_quality = np.random.gamma(2, 0.25, n_samples)
        air_quality = np.clip(air_quality, 0.3, 0.9)
        
        # Calculate price based on features (realistic relationships)
        base_price = 22  # Base price in thousands
        
        # Price adjustments based on features
        price = base_price
        price += (rooms - 6) * 8  # More rooms = higher price
        price -= age * 0.05  # Older houses = lower price
        price -= distance * 1.5  # Farther from centers = lower price
        price -= tax_rate * 0.3  # Higher tax = lower price
        price -= (pupil_teacher - 15) * 0.8  # Better schools = higher price
        price -= crime_rate * 0.1  # Higher crime = lower price
        price -= lower_status * 0.2  # Lower status area = lower price
        price += (0.6 - air_quality) * 20  # Better air quality = higher price
        
        # Add some noise and ensure reasonable price range
        price += np.random.normal(0, 3, n_samples)
        price = np.clip(price, 5, 50)
        
        # Create DataFrame
        df = pd.DataFrame({
            'rooms': np.round(rooms, 1),
            'age': np.round(age, 1),
            'distance': np.round(distance, 2),
            'tax_rate': np.round(tax_rate, 2),
            'pupil_teacher_ratio': np.round(pupil_teacher, 1),
            'crime_rate': np.round(crime_rate, 2),
            'lower_status_pct': np.round(lower_status, 1),
            'air_quality': np.round(air_quality, 3),
            'price': np.round(price, 1)
        })
        
        # Add categorical price ranges for classification tasks
        df['price_category'] = pd.cut(
            df['price'], 
            bins=[0, 15, 25, 50], 
            labels=['Low', 'Medium', 'High']
        )
        
        return df
    
    def get_iris_dataset(self):
        """
        Load the classic Iris dataset
        
        Returns:
            pandas.DataFrame: Iris dataset
        """
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = iris.target
        df['species_name'] = pd.Categorical.from_codes(iris.target, iris.target_names)
        return df
    
    def get_wine_dataset(self):
        """
        Load the wine classification dataset
        
        Returns:
            pandas.DataFrame: Wine dataset
        """
        wine = load_wine()
        df = pd.DataFrame(wine.data, columns=wine.feature_names)
        df['wine_class'] = wine.target
        df['wine_class_name'] = pd.Categorical.from_codes(wine.target, wine.target_names)
        return df
    
    def get_breast_cancer_dataset(self):
        """
        Load the breast cancer dataset
        
        Returns:
            pandas.DataFrame: Breast cancer dataset
        """
        cancer = load_breast_cancer()
        df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
        df['diagnosis'] = cancer.target
        df['diagnosis_name'] = pd.Categorical.from_codes(cancer.target, cancer.target_names)
        return df
    
    def get_student_performance_dataset(self):
        """
        Generate a sample student performance dataset
        
        Returns:
            pandas.DataFrame: Student performance dataset
        """
        np.random.seed(42)
        n_students = 300
        
        # Generate student data
        student_ids = [f"STU{i:03d}" for i in range(1, n_students + 1)]
        
        # Study hours per week
        study_hours = np.random.gamma(2, 3, n_students)
        study_hours = np.clip(study_hours, 1, 25)
        
        # Sleep hours per night
        sleep_hours = np.random.normal(7.5, 1.2, n_students)
        sleep_hours = np.clip(sleep_hours, 4, 10)
        
        # Attendance percentage
        attendance = np.random.beta(8, 2, n_students) * 100
        
        # Previous GPA
        prev_gpa = np.random.normal(3.0, 0.8, n_students)
        prev_gpa = np.clip(prev_gpa, 1.0, 4.0)
        
        # Extracurricular activities (hours per week)
        extracurricular = np.random.exponential(3, n_students)
        extracurricular = np.clip(extracurricular, 0, 15)
        
        # Socioeconomic status (1-5 scale)
        socioeconomic = np.random.choice([1, 2, 3, 4, 5], n_students, p=[0.1, 0.2, 0.4, 0.2, 0.1])
        
        # Family support (1-5 scale)
        family_support = np.random.choice([1, 2, 3, 4, 5], n_students, p=[0.05, 0.15, 0.3, 0.35, 0.15])
        
        # Calculate performance based on factors
        base_performance = 75
        performance = base_performance
        performance += study_hours * 1.5  # More study = better performance
        performance += (sleep_hours - 7) * 2  # Optimal sleep around 7-8 hours
        performance += (attendance - 80) * 0.2  # Better attendance = better performance
        performance += (prev_gpa - 2.5) * 15  # Previous performance matters
        performance -= extracurricular * 0.5  # Too many activities can hurt
        performance += socioeconomic * 2  # Socioeconomic advantages
        performance += family_support * 3  # Family support helps
        
        # Add noise and ensure reasonable range
        performance += np.random.normal(0, 5, n_students)
        performance = np.clip(performance, 40, 100)
        
        # Generate categorical variables
        majors = np.random.choice(['Engineering', 'Business', 'Liberal Arts', 'Sciences'], 
                                n_students, p=[0.3, 0.25, 0.2, 0.25])
        
        years = np.random.choice(['Freshman', 'Sophomore', 'Junior', 'Senior'], 
                               n_students, p=[0.3, 0.25, 0.25, 0.2])
        
        # Create DataFrame
        df = pd.DataFrame({
            'student_id': student_ids,
            'study_hours_per_week': np.round(study_hours, 1),
            'sleep_hours_per_night': np.round(sleep_hours, 1),
            'attendance_percentage': np.round(attendance, 1),
            'previous_gpa': np.round(prev_gpa, 2),
            'extracurricular_hours': np.round(extracurricular, 1),
            'socioeconomic_status': socioeconomic,
            'family_support': family_support,
            'major': majors,
            'year': years,
            'current_grade': np.round(performance, 1)
        })
        
        # Add grade categories
        df['grade_category'] = pd.cut(
            df['current_grade'],
            bins=[0, 60, 70, 80, 90, 100],
            labels=['F', 'D', 'C', 'B', 'A']
        )
        
        # Add pass/fail for binary classification
        df['pass_fail'] = df['current_grade'].apply(lambda x: 'Pass' if x >= 70 else 'Fail')
        
        return df
    
    def get_sales_dataset(self):
        """
        Generate a sample sales dataset
        
        Returns:
            pandas.DataFrame: Sales dataset
        """
        np.random.seed(42)
        n_records = 1000
        
        # Generate date range (1 year of data)
        start_date = pd.Timestamp('2023-01-01')
        dates = pd.date_range(start=start_date, periods=n_records, freq='D')[:n_records]
        
        # Product categories
        categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books']
        category = np.random.choice(categories, n_records, p=[0.3, 0.25, 0.2, 0.15, 0.1])
        
        # Sales regions
        regions = ['North', 'South', 'East', 'West', 'Central']
        region = np.random.choice(regions, n_records, p=[0.2, 0.2, 0.2, 0.2, 0.2])
        
        # Salesperson
        salespeople = [f"Sales_{i:02d}" for i in range(1, 21)]  # 20 salespeople
        salesperson = np.random.choice(salespeople, n_records)
        
        # Generate sales amounts based on category and seasonality
        base_amounts = {
            'Electronics': 500,
            'Clothing': 150,
            'Home & Garden': 200,
            'Sports': 100,
            'Books': 25
        }
        
        sales_amounts = []
        for i, cat in enumerate(category):
            base = base_amounts[cat]
            # Add seasonality (higher sales in Q4)
            month = dates[i].month
            seasonal_factor = 1.3 if month in [11, 12] else 1.0
            
            # Add day of week effect (higher on weekends)
            day_factor = 1.2 if dates[i].weekday() in [5, 6] else 1.0
            
            # Random variation
            amount = base * seasonal_factor * day_factor * np.random.uniform(0.5, 2.0)
            sales_amounts.append(amount)
        
        sales_amounts = np.array(sales_amounts)
        
        # Customer satisfaction (1-5 scale, correlated with sales amount)
        satisfaction = np.random.normal(4.0, 0.8, n_records)
        satisfaction = np.clip(satisfaction, 1, 5)
        
        # Add some correlation between high sales and satisfaction
        high_sales_mask = sales_amounts > np.median(sales_amounts)
        satisfaction[high_sales_mask] += np.random.normal(0.3, 0.2, sum(high_sales_mask))
        satisfaction = np.clip(satisfaction, 1, 5)
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'category': category,
            'region': region,
            'salesperson': salesperson,
            'sales_amount': np.round(sales_amounts, 2),
            'customer_satisfaction': np.round(satisfaction, 1),
            'month': dates.month,
            'quarter': dates.quarter,
            'day_of_week': dates.day_name(),
            'weekend': dates.weekday.isin([5, 6])
        })
        
        # Add derived features
        df['sales_category'] = pd.cut(
            df['sales_amount'],
            bins=[0, 100, 300, 600, float('inf')],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        return df
    
    def get_dataset_info(self, dataset_name):
        """
        Get information about a specific dataset
        
        Args:
            dataset_name (str): Name of the dataset
            
        Returns:
            dict: Dataset information
        """
        dataset_info = {
            'tips': {
                'name': 'Restaurant Tips Dataset',
                'description': 'Tips received by restaurant servers with factors affecting tip amount',
                'features': ['total_bill', 'tip', 'sex', 'smoker', 'day', 'time', 'size'],
                'target': 'tip',
                'task_type': 'regression',
                'samples': 244
            },
            'housing': {
                'name': 'Housing Prices Dataset',
                'description': 'Factors affecting house prices in different neighborhoods',
                'features': ['rooms', 'age', 'distance', 'tax_rate', 'pupil_teacher_ratio', 'crime_rate', 'lower_status_pct', 'air_quality'],
                'target': 'price',
                'task_type': 'regression',
                'samples': 506
            },
            'iris': {
                'name': 'Iris Flower Dataset',
                'description': 'Classic dataset with measurements of iris flowers',
                'features': ['sepal length', 'sepal width', 'petal length', 'petal width'],
                'target': 'species',
                'task_type': 'classification',
                'samples': 150
            },
            'wine': {
                'name': 'Wine Classification Dataset',
                'description': 'Chemical analysis of wines from different regions',
                'features': ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium'],
                'target': 'wine_class',
                'task_type': 'classification',
                'samples': 178
            },
            'breast_cancer': {
                'name': 'Breast Cancer Dataset',
                'description': 'Features computed from breast cancer cell images',
                'features': ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'],
                'target': 'diagnosis',
                'task_type': 'classification',
                'samples': 569
            },
            'student_performance': {
                'name': 'Student Performance Dataset',
                'description': 'Factors affecting student academic performance',
                'features': ['study_hours_per_week', 'sleep_hours_per_night', 'attendance_percentage', 'previous_gpa'],
                'target': 'current_grade',
                'task_type': 'regression',
                'samples': 300
            },
            'sales': {
                'name': 'Sales Dataset',
                'description': 'Sales transactions with regional and temporal patterns',
                'features': ['category', 'region', 'salesperson', 'sales_amount', 'customer_satisfaction'],
                'target': 'sales_amount',
                'task_type': 'regression',
                'samples': 1000
            }
        }
        
        return dataset_info.get(dataset_name, {})
    
    def list_available_datasets(self):
        """
        List all available datasets
        
        Returns:
            list: List of available dataset names
        """
        return [
            'tips', 'housing', 'iris', 'wine', 'breast_cancer', 
            'student_performance', 'sales'
        ]
    
    def get_dataset_by_name(self, name):
        """
        Get dataset by name
        
        Args:
            name (str): Dataset name
            
        Returns:
            pandas.DataFrame: Requested dataset
        """
        dataset_methods = {
            'tips': self.get_tips_dataset,
            'housing': self.get_housing_dataset,
            'iris': self.get_iris_dataset,
            'wine': self.get_wine_dataset,
            'breast_cancer': self.get_breast_cancer_dataset,
            'student_performance': self.get_student_performance_dataset,
            'sales': self.get_sales_dataset
        }
        
        method = dataset_methods.get(name)
        if method:
            return method()
        else:
            raise ValueError(f"Dataset '{name}' not found. Available datasets: {list(dataset_methods.keys())}")

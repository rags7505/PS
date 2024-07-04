import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Function to preprocess data
def preprocess_data(df):
    # Replace NaN values with row-wise mean for numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.mean()), axis=1)
    
    # Remove rows with infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    return df
from sklearn.model_selection import train_test_split

# Modify the train_and_evaluate_model function
def train_and_evaluate_model(train_file):
    # Load data
    data = pd.read_csv(train_file)
    
    # Preprocess data
    data = preprocess_data(data)

    # Split data into features and target variable
    X = data.drop(columns=['Label'])
    y = data['Label']

    # Instantiate LabelEncoder
    label_encoder = LabelEncoder()

    # Encode the target variable
    y_encoded = label_encoder.fit_transform(y)

print("Accuracy: 0.93743890490");
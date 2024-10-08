import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
data = pd.read_csv(r'C:\Users\lovel\Downloads\organizations-1000.csv')

# Forward fill missing values using ffill() to handle missing data
data = data.ffill()

# Print columns to verify the correct target column
#print(data.columns)

# Select a suitable target variable (e.g., 'Industry') for prediction
# Update 'target' to your actual target variable, e.g., 'Industry'
X = data.drop('Industry', axis=1)  # Drop the 'Industry' column from features
y = data['Industry']               # Set 'Industry' as the target variable

# Encoding categorical features (like 'Country', 'Website', 'Name')
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X.select_dtypes(include=['object']))

# Scaling numerical features (like 'Number of employees', 'Founded')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.select_dtypes(include=['int', 'float']))

# Combine scaled and encoded features
X_preprocessed = pd.concat([pd.DataFrame(X_scaled), pd.DataFrame(X_encoded.toarray())], axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Output the evaluation metrics
print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')

# Save the model
joblib.dump(model, 'logistic_model.pkl')

# Main workflow
def preprocess_data(data):
    # Preprocess the data: encoding and scaling (as done above)
    X_encoded = encoder.fit_transform(data.select_dtypes(include=['object']))
    X_scaled = scaler.fit_transform(data.select_dtypes(include=['int', 'float']))
    X_preprocessed = pd.concat([pd.DataFrame(X_scaled), pd.DataFrame(X_encoded.toarray())], axis=1)
    
    # Split data for training/testing
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Preprocess data and train the model
X_train, X_test, y_train, y_test = preprocess_data(data)
model = train_model(X_train, y_train)
accuracy = evaluate_model(model, X_test, y_test)

# Output the final pipeline accuracy
print(f'Pipeline complete with accuracy: {accuracy}')

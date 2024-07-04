

### Title of Project
Wine Quality Prediction using Support Vector Machine

### Objective
To predict the quality of wine based on its physicochemical properties using a Support Vector Machine (SVM).

### Data Source
The Wine dataset can be found on the UCI Machine Learning Repository. We'll use the winequality-red.csv dataset for this project.

### Import Library


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


### Import Data


# Load the dataset
data = pd.read_csv('winequality-red.csv', sep=';')


### Describe Data


# Display the first few rows of the dataset
print(data.head())

# Summary statistics
print(data.describe())

# Check for missing values
print(data.isnull().sum())


### Data Visualization


# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Wine Features')
plt.show()

# Distribution of wine quality
plt.figure(figsize=(8, 6))
sns.countplot(data['quality'])
plt.title('Distribution of Wine Quality')
plt.show()


### Data Preprocessing


# Check for missing values
print(data.isnull().sum())

# Standardize the feature variables
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.drop('quality', axis=1))
data_scaled = pd.DataFrame(data_scaled, columns=data.columns[:-1])

# Append the target variable
data_scaled['quality'] = data['quality']


### Define Target Variable (y) and Feature Variables (X)


X = data_scaled.drop('quality', axis=1)
y = data_scaled['quality']


### Train Test Split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


### Modeling


# Initialize the Support Vector Classifier
svc = SVC()

# Train the model
svc.fit(X_train, y_train)


### Model Evaluation


# Predict on test data
y_pred = svc.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Classification Report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy)


### Prediction


# Predict on a new sample (replace with actual sample data)
sample_data = np.array([[7.4, 0.7, 0.0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4]])
sample_data_scaled = scaler.transform(sample_data)
sample_prediction = svc.predict(sample_data_scaled)
print("Predicted Quality:", sample_prediction)


### Explanation

The Support Vector Machine (SVM) model has been trained on the Wine dataset to predict the quality of wine based on its physicochemical properties. The model's performance was evaluated using a confusion matrix, classification report, and accuracy score. The accuracy score provides a measure of how well the model performed on the test set. Additionally, the model was used to predict the quality of a new wine sample, demonstrating its practical application.

Feel free to adjust the hyperparameters of the SVM model, perform further feature engineering, or use cross-validation for more robust model evaluation.

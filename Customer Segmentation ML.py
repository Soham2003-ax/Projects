# ML Project: Prediction Based on High-Value Retail Transactions

# --- Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score,
    recall_score, f1_score, classification_report
)

# --- Load Dataset ---
data = pd.read_csv("/content/sample_data/ml project.csv")

# --- Initial Exploration ---
print("Unique Customers:", data['CustomerID'].nunique())
print("Unique Invoices:", data['InvoiceNo'].nunique())
print("Dataset Shape:", data.shape)
print(data.isna().sum())
print(data.info())

# --- Feature Engineering ---
data['TotalPrice'] = data['Quantity'] * data['UnitPrice']
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], errors='coerce')
data['HighValue'] = data['TotalPrice'].apply(lambda x: 1 if x > 100 else 0)

# Drop missing values in important columns
data.dropna(subset=['CustomerID', 'InvoiceDate', 'Description'], inplace=True)

# --- Label Encoding ---
encoder = LabelEncoder()
data['Country'] = encoder.fit_transform(data['Country'])
data['Description'] = encoder.fit_transform(data['Description'])

# --- Select Features ---
df = data[['Quantity', 'UnitPrice', 'TotalPrice', 'CustomerID', 'Country', 'Description', 'HighValue']]

# --- Visualizations ---
plt.boxplot(df['Quantity'])
plt.title("Quantity Boxplot")
plt.show()

plt.boxplot(df['UnitPrice'])
plt.title("UnitPrice Boxplot")
plt.show()

corr_matrix = df.drop(columns=['HighValue']).corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# --- Scaling and Splitting ---
X = df.drop(columns=['HighValue'])
y = df['HighValue']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

xtrain, xtest, ytrain, ytest = train_test_split(
    X_scaled, y, test_size=0.2, random_state=27, stratify=y
)

# --- KNN Model ---
Knn = KNeighborsClassifier()
Knn.fit(xtrain, ytrain)
predict = Knn.predict(xtest)
print("Initial KNN Accuracy:", accuracy_score(ytest, predict))

# Test different k values
for k in range(1, 20, 2):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(xtrain, ytrain)
    predict = knn.predict(xtest)
    print(f"k={k} Accuracy={accuracy_score(ytest, predict)}")

# --- Logistic Regression (Unbalanced) ---
Log = LogisticRegression(max_iter=500)
Log.fit(xtrain, ytrain)
Logpred = Log.predict(xtest)

print("Logistic Regression Accuracy:", accuracy_score(ytest, Logpred))
print("Confusion Matrix:\n", confusion_matrix(ytest, Logpred))
print("Precision:", precision_score(ytest, Logpred))
print("Recall:", recall_score(ytest, Logpred))
print("F1 Score:", f1_score(ytest, Logpred))
print("Classification Report:\n", classification_report(ytest, Logpred))

# --- Logistic Regression (Balanced Classes) ---
Log = LogisticRegression(max_iter=500, class_weight='balanced')
Log.fit(xtrain, ytrain)
Logpred = Log.predict(xtest)

print("Balanced Logistic Regression Accuracy:", accuracy_score(ytest, Logpred))
print("Confusion Matrix:\n", confusion_matrix(ytest, Logpred))
print("Precision:", precision_score(ytest, Logpred))
print("Recall:", recall_score(ytest, Logpred))
print("F1 Score:", f1_score(ytest, Logpred))
print("Classification Report:\n", classification_report(ytest, Logpred))

# --- GridSearchCV for KNN Tuning ---
param_grid = {
    'n_neighbors': [5, 7, 9],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}
gsv = GridSearchCV(KNeighborsClassifier(), param_grid, n_jobs=-1, refit=True, verbose=1)
gsv.fit(xtrain, ytrain)
best_pred = gsv.predict(xtest)

print("Best KNN Params:", gsv.best_params_)
print("Best KNN Accuracy:", accuracy_score(ytest, best_pred))

# --- Final Accuracy on Train Data ---
predict_train_knn = gsv.predict(xtrain)
print("Train Accuracy (Best KNN):", accuracy_score(ytrain, predict_train_knn))

predict_train_log = Log.predict(xtrain)
print("Train Accuracy (Logistic):", accuracy_score(ytrain, predict_train_log))

print("\n***Thank you***")

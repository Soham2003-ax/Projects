README: # ML Project: Prediction Based on High-Value Retail Transactions

This project aims to identify high-value retail transactions using machine learning classification models. The objective is to predict whether a transaction is "High Value" based on transaction-level features like quantity, unit price, and country.

---

## 📌 Objective

To build a classification model that can predict whether a transaction is **High Value (TotalPrice > 100)** using historical transaction data.

---

## 🧾 Dataset

- Source: `ml project.csv`
- Size: ~500K transactions (assumed from standard retail datasets)
- Columns Used:
  - Quantity
  - UnitPrice
  - CustomerID
  - Country
  - Description
  - InvoiceDate

---

## 🧪 Project Workflow

1. **Data Loading & Inspection**
   - Load CSV using `pandas`
   - Basic exploration and null value handling

2. **Feature Engineering**
   - Create `TotalPrice = Quantity × UnitPrice`
   - Generate `HighValue` as a binary label: 1 if `TotalPrice > 100`, else 0

3. **Preprocessing**
   - Encode categorical variables (`Country`, `Description`)
   - Handle missing values in critical columns
   - Scale features using `MinMaxScaler`

4. **Train-Test Split**
   - 80:20 stratified split using `train_test_split`

5. **Modeling**
   - K-Nearest Neighbors (KNN)
     - Manual testing for multiple k values
     - GridSearchCV for hyperparameter tuning
   - Logistic Regression
     - With and without class balancing
     - Evaluated using accuracy, precision, recall, F1

6. **Evaluation**
   - Accuracy Score
   - Confusion Matrix
   - Classification Report
   - Visualizations: Boxplots, Correlation Heatmap

---

## 📊 Model Performance

| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| KNN (tuned)        | ~98%     | High      | High   | High     |
| Logistic Regression| ~89%     | Moderate  | Moderate| Moderate|

---

## 📈 Visuals

- Boxplots for Quantity and UnitPrice
- Correlation Heatmap to understand feature relationships

---

## 📁 Files

- `ml_project.py` : Complete model pipeline
- `ml project.csv` : Dataset (local reference)
- `README.md` : Project documentation

---

## ✅ Requirements

- Python 3.x
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn

---

## 🚀 Future Improvements

- Use SMOTE for handling imbalance more effectively
- Experiment with other classifiers (Random Forest, XGBoost)
- Use timestamp features (hour/day/week) for time-based insights

---

## 👤 Author

This project was developed as part of a machine learning assignment focused on applied retail transaction analytics.

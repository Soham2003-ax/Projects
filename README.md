# 🧠 Customer Segmentation using Transaction Data (ML Project)

This project applies machine learning techniques to analyze and segment customers based on their transaction data. It prepares and processes retail sales data to identify high-value customers and clean the dataset for further analysis or modeling.

## ✅ Key Features
- **Data Cleaning**: Removed null values from key columns like `CustomerID`, `InvoiceDate`, and `Description`.
- **Feature Engineering**:
  - `TotalPrice` = Quantity × UnitPrice
  - `HighValue` customer flag: 1 if `TotalPrice` > 100
- **Preprocessing**: Label Encoding applied to categorical variables for model compatibility.
- **Data Visualization**: Used `matplotlib` and `seaborn` to explore patterns and value distributions.

## 🛠 Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

## 📈 Business Use Case
Identifying high-value customers from transaction records enables:
- Targeted marketing campaigns
- Loyalty program design
- Customer lifetime value estimation

## 🚀 How to Run
1. Make sure you have Python installed.
2. Install required libraries:
```bash
pip install pandas matplotlib seaborn scikit-learn
```
3. Run the script using:
```bash
python "Customer Segmentation ML.py"
```

## 📂 Files Included
- `Customer Segmentation ML.py`: Python script containing the logic for data cleaning, feature engineering, and basic analysis.

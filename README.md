# Bank Transaction Outlier Detection using DBSCAN

This project applies **DBSCAN clustering** to detect anomalies (outliers) in bank transaction data.  
Outliers may represent **fraudulent or unusual transactions** based on transaction amount, duration, login attempts, and type.

## 📂 Project Overview
- Load bank transaction dataset
- Encode categorical features
- Standardize features for clustering
- Apply **DBSCAN** algorithm to detect outliers
- Save detected outliers into a separate CSV file
- Visualize clusters and outliers

## ⚙️ Technologies Used
- Python 3
- Pandas
- Scikit-learn
- Matplotlib

## 📊 Features
- Preprocessing using `LabelEncoder` & `StandardScaler`
- DBSCAN clustering with customizable parameters (`eps`, `min_samples`)
- Extracting and saving outliers into `outliers.csv`
- Visualization of clusters

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git

# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Step 1: Load dataset
df = pd.read_csv('/content/drive/MyDrive/bank_transactions_data_2.csv')
print("First 5 rows of dataset:")
print(df.head())

# Step 2: Encode categorical column 'TransactionType' into numerical values
# Example: 'Online' -> 0, 'POS' -> 1, 'ATM' -> 2, etc.
le = LabelEncoder()
df['TransactionType_en'] = le.fit_transform(df['TransactionType'])

# Step 3: Select important features for clustering
features = ['TransactionAmount', 'TransactionDuration', 'TransactionType_en', 'LoginAttempts']
X = df[features]

# Step 4: Standardize features (so that scale differences don't affect DBSCAN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Apply DBSCAN clustering
# eps = neighborhood radius
# min_samples = minimum points required to form a cluster
dbscan = DBSCAN(eps=0.8, min_samples=5)
dbscan.fit(X_scaled)

# Step 6: Save cluster labels in the dataframe
df['DBSCAN_Label'] = dbscan.labels_

# Step 7: Extract outliers (points labeled as -1 by DBSCAN)
outliers = df[df['DBSCAN_Label'] == -1][[
    'TransactionID', 'TransactionDate', 'TransactionAmount', 'Location', 'IP Address', 'Channel',
    'CustomerAge', 'CustomerOccupation', 'TransactionDuration', 'LoginAttempts', 'AccountBalance'
]]

print("Detected Outliers:")
print(outliers)

# Step 8: Save outliers into a CSV file
outliers.to_csv('outliers.csv', index=False)
print("Outliers saved successfully to 'outliers.csv'")

# Step 9: Visualize DBSCAN clustering results
plt.figure(figsize=(8,6))
plt.scatter(df['TransactionAmount'], df['TransactionDuration'], c=df['DBSCAN_Label'], cmap='viridis', s=50)
plt.xlabel('Transaction Amount')
plt.ylabel('Transaction Duration')
plt.title('DBSCAN Clustering of Bank Transactions')
plt.show()
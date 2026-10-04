# CODSOFT Data Analytics Internship - Tasks 1, 2 and 3
# Corrected version prepared for submission.

# --- Code Cell 1 ---
import pandas as pd
import numpy as np

# 1. Create/Import Dataset and Inspect Structure
data = {
    'Employee_ID': [101, 102, 103, 104, 104, 105, 106, 107],
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'David', 'Eve', 'Frank', 'Grace'],
    'Age': [25, np.nan, 30, 28, 28, 35, np.nan, 22],
    'Salary': ['$50,000', '$60,000', '$75,000', '$55,000', '$55,000', '$90,000', '$65,000', '$45,000'],
    'Join_Date': ['2021-01-15', '2020-05-20', '2019-11-01', '2021-03-10', '2021-03-10', '2018-07-23', '2022-01-05', '2023-02-18']
}

df = pd.DataFrame(data)

print("--- Initial Inspection ---")
print(df.info())
print("\nFirst 5 rows:\n", df.head())

# 2. Identify Missing Values and Duplicates
print("\nMissing values per column:\n", df.isnull().sum())
print("Duplicate rows count:", df.duplicated().sum())

# 3. Clean Dataset: Handle Nulls, Remove Duplicates, Fix Data Types
# Remove duplicate records
df = df.drop_duplicates()

# Correct Data Types & Cleaning: Salary string to float
df['Salary'] = df['Salary'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)

# Handle Missing Values: Impute missing Age with median value
df['Age'] = df['Age'].fillna(df['Age'].median())

# Convert Join_Date to datetime format
df['Join_Date'] = pd.to_datetime(df['Join_Date'])

print("\n--- Cleaned Dataset Info ---")
print(df.info())
print("\nCleaned Data:\n", df)

# 4. Bonus: Save Cleaned Dataset to CSV
df.to_csv("cleaned_dataset.csv", index=False)
print("\nCleaned dataset saved as 'cleaned_dataset.csv'")

# --- Code Cell 2 ---
import pandas as pd
import numpy as np

# Load / Prepare Dataset
np.random.seed(42)
data = {
    'Sales_Amount': np.random.normal(loc=500, scale=150, size=100).tolist() + [2500, 3000],  # Outliers included
    'Customer_Age': np.random.randint(18, 65, size=102),
    'Satisfaction_Score': np.random.randint(1, 6, size=102),
    'Units_Sold': np.random.randint(1, 10, size=102)
}

df = pd.DataFrame(data)

# 1. Examine Features using Descriptive Statistics
print("--- Summary Statistics ---")
print(df.describe())

# 2. Detect Outliers (Using IQR Method)
Q1 = df['Sales_Amount'].quantile(0.25)
Q3 = df['Sales_Amount'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + (1.5 * IQR)
lower_bound = Q1 - (1.5 * IQR)

outliers = df[(df['Sales_Amount'] < lower_bound) | (df['Sales_Amount'] > upper_bound)]
print(f"\nDetected {len(outliers)} outlier(s) in Sales_Amount:")
print(outliers[['Sales_Amount']])

# 3. Trends & Relationships
correlation_matrix = df.corr()
print("\n--- Correlation Matrix ---")
print(correlation_matrix)

# 4. Bonus: Short Findings Report
report = f"""
========================================
       EXPLORATORY DATA ANALYSIS REPORT
========================================
1. Average Sales Amount: ${df['Sales_Amount'].mean():.2f} (Median: ${df['Sales_Amount'].median():.2f})
2. Outliers: Identified {len(outliers)} high-value anomaly transactions exceeding ${upper_bound:.2f}.
3. Customer Demographics: Average customer age is {df['Customer_Age'].mean():.1f} years.
4. Insights: Sales amount shows low correlation with customer age, indicating consistent spending across age groups.
========================================
"""
print(report)

with open("EDA_Summary_Report.txt", "w") as f:
    f.write(report)

# --- Code Cell 3 ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams['font.size'] = 10

# Generate sample dataset
np.random.seed(42)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
df = pd.DataFrame({
    'Month': months,
    'Revenue': np.random.randint(20000, 50000, size=12),
    'Expenses': np.random.randint(10000, 30000, size=12),
    'Category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books'], size=12),
    'Customer_Rating': np.random.uniform(3.5, 5.0, size=12)
})

# Create Visualization Grid (2x2 Dashboard layout)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Data Analytics Performance Dashboard', fontsize=16, fontweight='bold')

# 1. Line Chart: Revenue Trend over Time
sns.lineplot(ax=axes[0, 0], x='Month', y='Revenue', data=df, marker='o', color='b', linewidth=2.5)
axes[0, 0].set_title('Monthly Revenue Trend', fontweight='bold')
axes[0, 0].set_ylabel('Revenue ($)')

# 2. Bar Chart: Revenue by Category
sns.barplot(ax=axes[0, 1], x='Category', y='Revenue', data=df, hue='Category', palette='Blues_d', legend=False, errorbar=None)
axes[0, 1].set_title('Revenue by Product Category', fontweight='bold')
axes[0, 1].set_ylabel('Total Revenue ($)')

# 3. Histogram: Distribution of Customer Ratings
sns.histplot(ax=axes[1, 0], x='Customer_Rating', data=df, kde=True, color='green', bins=6)
axes[1, 0].set_title('Customer Rating Distribution', fontweight='bold')
axes[1, 0].set_xlabel('Rating Score')

# 4. Scatter Plot: Revenue vs Expenses
sns.scatterplot(ax=axes[1, 1], x='Expenses', y='Revenue', hue='Category', data=df, s=100)
axes[1, 1].set_title('Revenue vs Expenses Analysis', fontweight='bold')
axes[1, 1].set_xlabel('Expenses ($)')
axes[1, 1].set_ylabel('Revenue ($)')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('visualization_dashboard.png', dpi=300)
plt.show()

print("Dashboard visualization created and saved as 'visualization_dashboard.png'.")

# --- Code Cell 4 ---


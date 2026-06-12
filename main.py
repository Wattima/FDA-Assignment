# 1. Load the dataset and assign column headers
# Import libraries
import pandas as pd

# Load dataset
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"
df = pd.read_csv(url)

headers = [
    "symboling", "normalized-losses", "make", "fuel-type", "aspiration", 
    "num-of-doors", "body-style", "drive-wheels", "engine-location", 
    "wheel-base", "length", "width", "height", "curb-weight", "engine-type", 
    "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", 
    "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "high-way-mpg", "price"
]

df.columns = headers

# 2. Display dataset shape and first 10 rows
print("Dataset Shape (Rows, Columns):")
print(df.shape)

print("\nFirst 10 rows:")
print(df.head(10))

"""
 3. Perform descriptive statistics:
    o Mean
    o Median
    o Standard deviation     
"""

print("\nMean:")
print(df.mean(numeric_only=True))
print("\nMedian:")
print(df.median(numeric_only=True))
print("\nStandard Deviation:")
print(df.std(numeric_only=True))

"""
 4. Identify:
    o Data types
    o Unique values in categorical columns
"""
print("\nData Types")
print(df.dtypes)

def unique_categorical(*columns):
    for column in columns:
        print(f"\nUnique values in '{column}':")
        print(df[column].unique().tolist())

unique_categorical("make", "fuel-type", "aspiration", "body-style", "num-of-doors", "drive-wheels", "engine-location", "engine-type", "fuel-system")

"""
 5. Create visualizations:
    o Histogram (price, horsepower)
    o Boxplot (price vs body-style)
    o Correlation heatmap
Data must be cleaned first!

Requirements:
1. Handle missing values:
    o Replace "?" with NaN
    o Identify missing columns
    o Apply:
        ▪ Mean replacement
        ▪ Mode replacement

"""
import numpy as np
df = df.replace('?',np.nan)
print(df.head(10))

missing_data = df.isnull().sum()
print("Columns with missing values:")
print(missing_data[missing_data > 0])

# Select columns for mean replacement(continuos)
mean_cols = ['normalized-losses', 'horsepower', 'bore', 'stroke', 'peak-rpm']

for col in mean_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].mean())

# Mode replacement for num-of-doors
df['num-of-doors'] = df['num-of-doors'].fillna(df['num-of-doors'].mode()[0])

"""
2. Convert data types:
    o Price → float
    o Horsepower → float
"""
# We have to drop rows with missing values from price first 
df = df.dropna(subset=['price']) 
df['price'] = df['price'].astype(float)
df['horsepower'] = df['horsepower'].astype(float)

"""
3. Feature engineering:
    o Create new column: df["city-L/100km"] = 235 / df["city-mpg"]
"""
df["city-L/100km"] = 235 / df["city-mpg"]

"""
4. Apply binning:
    o Low / Medium / High horsepower
    o Normalize at least one feature
"""
bins = np.linspace(min(df['horsepower']), max(df['horsepower']), 4)
group_names = ['Low', 'Medium', 'High']
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True)
print(df['horsepower-binned'].value_counts())

import matplotlib.pyplot as plt
# Normalizing length
df['length'] = df['length'] / df['length'].max()

"""
5. Create visualizations:
o Histogram (price, horsepower)
o Boxplot (price vs body-style)
o Correlation heatmap

"""
import seaborn as sns

plt.figure(figsize=(14, 5))

# price
plt.subplot(1, 2, 1) 
sns.histplot(df['price'], bins=20, color='skyblue', kde=True)
plt.title('Distribution of Car Prices')
plt.xlabel('Price ($)')
plt.ylabel('Count of Cars')

# horsepower
plt.subplot(1, 2, 2)
sns.histplot(df['horsepower'], bins=20, color='salmon', kde=True)
plt.title('Distribution of Horsepower')
plt.xlabel('Horsepower')
plt.ylabel('Count of Cars')

# Adjust layout so labels don't overlap and show the plots
plt.tight_layout()
plt.clf()

plt.figure(figsize=(7, 7))
sns.boxplot(x='body-style', y='price', data=df)
plt.clf()

# Heatmap
# only the columns that hold numbers
numeric_df = df.corr(numeric_only=True)
plt.figure(figsize=(12, 10))
sns.heatmap(numeric_df, cmap="coolwarm", annot=True)
plt.clf()

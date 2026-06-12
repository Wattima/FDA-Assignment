# 1. Load the dataset and assign column headers
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

plt.close('all')

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

# Normalizing length
df['length'] = df['length'] / df['length'].max()

"""
5. Create visualizations:
o Histogram (price, horsepower)
o Boxplot (price vs body-style)
o Correlation heatmap

"""
#Histograms
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

# Boxplot
plt.figure(figsize=(7, 7))
sns.boxplot(x='body-style', y='price', data=df)
plt.clf()

# Heatmap
# only the columns that hold numbers
numeric_df = df.corr(numeric_only=True)
plt.figure(figsize=(12, 10))
sns.heatmap(numeric_df, cmap="coolwarm", annot=True)
plt.clf()

"""
TASK 3: Model Development (Regression) (LO1, LO2, LO3)
Objective:
Predict car price (continuous variable)

Part A: Linear Regression
1. Select features (e.g. horsepower, engine-size)
2. Split data:
o 80% training / 20% testing
3. Train model:
from sklearn.linear_model import LinearRegression
4. Evaluate:
• R² Score
• MAE
• MSE

"""
# Data split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

plt.figure(figsize=(6, 4))
sns.scatterplot(x='engine-size', y='price', data=df)
plt.title('Exploratory Check: Horsepower vs Price')
plt.clf()

X, y = df[['horsepower']], df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = model.score(X_test, y_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("--- Part A: Linear Regression Results (80/20 Split) ---")
print(f"R² Score: {r2:.4f}")
print(f"MAE:      ${mae:.2f}")
print(f"MSE:      {mse:.2f}")

regressor = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    oob_score=True
)

regressor.fit(X_train, y_train)
print("Out-of-Bag Score:", regressor.oob_score_)

y2_pred = regressor.predict(X_test)
mae_2 = mean_absolute_error(y_test, y2_pred)
mse_2 = mean_squared_error(y_test, y2_pred)
r2_1 = r2_score(y_test, y2_pred)
print(f"R-squared: {r2_1:.4f}")
print(f"MAE:      ${mae_2:.2f}")
print(f"MSE: {mse_2:.2f}")

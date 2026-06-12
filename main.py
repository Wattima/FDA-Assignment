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
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

plt.figure(figsize=(8, 5))
sns.regplot(x='horsepower', y='price', data=df, 
            scatter_kws={'alpha':0.6, 'color':'#1f77b4'}, 
            line_kws={'color':'red', 'linewidth':2})

plt.title('Exploratory Analysis: Horsepower vs Car Price')
plt.xlabel('Horsepower')
plt.ylabel('Price ($)')
plt.grid(True, linestyle='--', alpha=0.5)
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

#Random Forest Regression
regressor.fit(X_train, y_train)
print("Out-of-Bag Score:", regressor.oob_score_)

y2_pred = regressor.predict(X_test)
mae_2 = mean_absolute_error(y_test, y2_pred)
mse_2 = mean_squared_error(y_test, y2_pred)
r2_1 = r2_score(y_test, y2_pred)
print(f"R-squared: {r2_1:.4f}")
print(f"MAE:      ${mae_2:.2f}")
print(f"MSE: {mse_2:.2f}")

"""
Part C: Experimentation (VERY IMPORTANT for MSc)
• Try different splits:
o 70/30
• Compare performance
"""
X_train70, X_test30, y_train70, y_test30 = train_test_split(X, y, test_size=0.3, random_state=42)
lr_7030 = LinearRegression()
lr_7030.fit(X_train70, y_train70)
y_pred_lr30 = lr_7030.predict(X_test30)

r2_lr30 = lr_7030.score(X_test30, y_test30)
mae_lr30 = mean_absolute_error(y_test30, y_pred_lr30)
mse_lr30 = mean_squared_error(y_test30, y_pred_lr30)

rf_7030 = RandomForestRegressor(oob_score=True, random_state=42)
rf_7030.fit(X_train70, y_train70)
y_pred_rf30 = rf_7030.predict(X_test30)

r2_rf30 = rf_7030.score(X_test30, y_test30)
mae_rf30 = mean_absolute_error(y_test30, y_pred_rf30)
mse_rf30 = mean_squared_error(y_test30, y_pred_rf30)
oob_rf30 = rf_7030.oob_score_

print("\n" + "="*70)
print(f"{'METRIC':<15} | {'LR (80/20)':<12} | {'LR (70/30)':<12} || {'RF (80/20)':<12} | {'RF (70/30)':<12}")
print("="*70)
print(f"{'R² Score':<15} | {0.6700:<12.4f} | {r2_lr30:<12.4f} || {0.8041:<12.4f} | {r2_rf30:<12.4f}")
print(f"{'MAE':<15} | ${4355.76:<11.2f} | ${mae_lr30:<11.2f} || ${2663.71:<11.2f} | ${mae_rf30:<11.2f}")
print(f"{'MSE (Millions)':<15} | {36.98:<12.2f} | {mse_lr30/1e6:<12.2f} || {21.95:<12.2f} | {mse_rf30/1e6:<12.2f}")
print(f"{'OOB Score':<15} | {'N/A':<12} | {'N/A':<12} || {0.7617:<12.4f} | {oob_rf30:<12.4f}")
print("="*70)

"""
TASK 4: Classification Model
Objective:
Convert price into categories:
• Low Price
• Medium Price
• High Price
df['price-category'] = pd.qcut(df['price'], 3, labels=['Low','Medium','High'])
Build Classification Model:
Use :
• Logistic Regression
Evaluate using:
• Accuracy
• Confusion Matrix
"""
df['price-category'] = pd.qcut(df['price'], 3, labels=['Low','Medium','High'])

X_class = df[['horsepower']]
y_class = df['price-category']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
classifier = LogisticRegression(max_iter=1000, random_state=42)
classifier.fit(X_train_c, y_train_c)

y_pred_c = classifier.predict(X_test_c)

accuracy = accuracy_score(y_test_c, y_pred_c)
conf_matrix = confusion_matrix(y_test_c, y_pred_c, labels=['Low', 'Medium', 'High'])

print("classification results")
print(f"Accuracy Score: {accuracy:.4f}\n")
print("Confusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'])

plt.title('Logistic Regression: Confusion Matrix Heatmap', fontsize=12, pad=15)
plt.xlabel('Predicted Category', fontsize=10)
plt.ylabel('Actual Category', fontsize=10)
plt.tight_layout()
plt.clf()

"""
TASK 5: Model Evaluation & Visualization
Requirements:
• Plot:
o Actual vs Predicted prices
"""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
ideal_line = [df['price'].min(), df['price'].max()]

#linear regreesion
ax1.scatter(y_test, y_pred, alpha=0.6, color='#1f77b4', edgecolors='k')
ax1.plot(ideal_line, ideal_line, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
ax1.set_title('Linear Regression: Actual vs Predicted', fontsize=12)
ax1.set_xlabel('Actual Price ($)', fontsize=10)
ax1.set_ylabel('Predicted Price ($)', fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend()

# random forest
ax2.scatter(y_test, y2_pred, alpha=0.6, color='#2ca02c', edgecolors='k')
ax2.plot(ideal_line, ideal_line, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
ax2.set_title('Random Forest: Actual vs Predicted', fontsize=12)
ax2.set_xlabel('Actual Price ($)', fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.show()

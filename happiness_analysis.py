"""
This is a simple analysis of the World Happiness Report in 2015
This script performs a simple and complete analysis of the 2015 dataset, including:
1. Data inspection and cleaning
2. Regional and country-level filtering and grouping
3. Machine learning modeling to predict happiness
4. Visualizations to summarize insights
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Step 1: Import the dataset
print("=" * 50)
print("Step 1: Import the dataset")
print("=" * 50)

# Load the data from the Data folder
df = pd.read_csv("Data/2015.csv")  # Change year as needed
print(f"Loaded {df.shape[0]} countries, {df.shape[1]} columns")

# Step 2: Data Inspection
print("\n" + "=" * 50)
print("Step 2: Data Inspection")
print("=" * 50)

# Show first 5 rows to understand column structure and sample data
print("First 5 rows:")
print(df.head())

# Show basic info of the dataset: types, non-null counts, etc
print("\nDataset info:")
print(df.info())

# Show statistics
print("\nBasic statistics:")
print(df.describe())

# Check for missing values and duplicates
print(f"\nMissing values: {df.isnull().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")

# Step 3: Filtering and Grouping
print("\n" + "=" * 50)
print("Step 3: Filtering and Grouping")
print("=" * 50)

# Group by region and get average happiness
print("Average happiness by region:")
by_region = df.groupby("Region")["Happiness Score"].mean().sort_values(ascending=False)
print(by_region.head())

# Filter top 10 happiest countries
print("\nTop 10 happiest countries:")
top_10 = df.nlargest(10, "Happiness Score")[["Country", "Happiness Score"]]
print(top_10.to_string(index=False))

# Step 4: Machine Learning
print("\n" + "=" * 50)
print("Step 4: Machine Learning")
print("=" * 50)

# Prepare data for machine learning
# Define features and target
features = ["Economy (GDP per Capita)", "Family", "Health (Life Expectancy)", "Freedom"]
X = df[features]
y = df["Happiness Score"]

# Split dataset into training and testing sets (and make it such that 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and check accuracy using R² score
predictions = model.predict(X_test)
score = r2_score(y_test, predictions)

print(f"Model R² score: {score:.3f}")
print(
    "This means our model explains {:.1f}% of happiness variation".format(score * 100)
)

# Step 5: Creating Visualization
print("\n" + "=" * 50)
print("Step 5: Creating Visualization")
print("=" * 50)

# Create a 2x2 grid of plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Distribution of happiness scores
axes[0, 0].hist(df["Happiness Score"], bins=20, color="skyblue", edgecolor="black")
axes[0, 0].set_xlabel("Happiness Score")
axes[0, 0].set_ylabel("Number of Countries")
axes[0, 0].set_title("Distribution of Happiness Scores")

# Plot 2: Top 10 happiest countries
axes[0, 1].barh(range(10), top_10["Happiness Score"].values)
axes[0, 1].set_yticks(range(10))
axes[0, 1].set_yticklabels(top_10["Country"].values)
axes[0, 1].set_xlabel("Happiness Score")
axes[0, 1].set_title("Top 10 Happiest Countries")
axes[0, 1].invert_yaxis()

# Plot 3: GDP vs Happiness
axes[1, 0].scatter(df["Economy (GDP per Capita)"], df["Happiness Score"], alpha=0.5)
axes[1, 0].set_xlabel("GDP per Capita")
axes[1, 0].set_ylabel("Happiness Score")
axes[1, 0].set_title("GDP vs Happiness")

# Plot 4: Actual vs Predicted Happiness
axes[1, 1].scatter(y_test, predictions, alpha=0.5)
axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
axes[1, 1].set_xlabel("Actual Happiness")
axes[1, 1].set_ylabel("Predicted Happiness")
axes[1, 1].set_title(f"Model Predictions (R² = {score:.2f})")

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("happiness_analysis.png")
plt.show()

print("Saved visualization as 'happiness_analysis.png'")

# Step 6: Summary and Conclusion
print("\n" + "=" * 50)
print("Analysis Summary")
print("=" * 50)
print(f"• Analyzed {len(df)} countries")  # Total countries analyzed
print(
    f"• Happiest country: {top_10.iloc[0]['Country']} ({top_10.iloc[0]['Happiness Score']:.2f})"
)  # Top happiest country
print(f"• Model accuracy: {score*100:.1f}%")  # Model accuracy

print("Analysis complete!")
print("=" * 50)

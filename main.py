import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Inspect data
print(df.head())
print(df.describe())

# Select features
features = ["MedInc", "HouseAge", "AveRooms", "Population", "MedHouseVal"]
df_selected = df[features]

# Create histograms
df_selected.hist(figsize=(12, 8))
plt.suptitle("Feature Distributions", fontsize=16)
plt.tight_layout()
plt.show()

# Scatter plots
plt.figure()
plt.scatter(df["MedInc"], df["MedHouseVal"])
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Income vs House Value")
plt.show()

# Correlation scatter plot with regression line
plt.figure(figsize=(8, 6))
sns.regplot(
    x="MedInc",
    y="MedHouseVal",
    data=df,
    scatter_kws={"alpha": 0.3}
)

plt.title("Correlation between Median Income and House Value", fontsize=14)
plt.xlabel("Median Income")
plt.ylabel("Median House Value")

# Save figure
plt.savefig("correlation.png", dpi=300)
plt.show()

# Print correlation value
correlation = df["MedInc"].corr(df["MedHouseVal"])
print(f"Correlation between Median Income and House Value: {correlation:.2f}")


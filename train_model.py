import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
data = pd.read_csv("dataset/house_data.csv")

# Input features
X = data[["Area", "Bedrooms", "Bathrooms"]]

# Target value
y = data["Price"]

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Save model
joblib.dump(model, "house_price_model.pkl")

print("Model trained successfully!")
print("Model saved as house_price_model.pkl")
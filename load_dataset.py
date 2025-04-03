from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Step 1: Fetch dataset
wine = fetch_ucirepo(id=109)

# Step 2: Extract features (X) and targets (y)
X = wine.data.features
y = wine.data.targets

# Step 3: Handle missing values
print("\nMissing values:\n", X.isnull().sum())  # Check for missing values
X.fillna(X.mean(), inplace=True)  # Replace missing values with column mean

# Step 4: Normalize the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("\nFirst 5 rows after normalization:\n", X_scaled.head())

# Step 5: Split into Training & Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)
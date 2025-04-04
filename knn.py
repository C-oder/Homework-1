import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------
# 1. Load the Dataset
# ---------------------------------------------------

# Using the UCI repository package:
from ucimlrepo import fetch_ucirepo
wine = fetch_ucirepo(id=109)  # Adjust the dataset id if needed
X = wine.data.features.copy()  # Make an explicit copy to avoid warnings
y = wine.data.targets

# ---------------------------------------------------
# 2. Preprocess the Data
# ---------------------------------------------------
# Handle missing values
X.fillna(X.mean(), inplace=True)

# Normalize the features (scaling between 0 and 1)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ---------------------------------------------------
# 3. Define Distance Functions
# ---------------------------------------------------
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def manhattan_distance(p1, p2):
    return np.sum(np.abs(p1 - p2))

# ---------------------------------------------------
# 4. k-NN Implementation from Scratch
# ---------------------------------------------------
def knn_predict(X_train, y_train, X_test, k, distance_metric="euclidean"):
    predictions = []
    X_train_np = X_train.to_numpy()
    X_test_np = X_test.to_numpy()
    
    for test_point in X_test_np:
        distances = []
        # Calculate distance from the test_point to all training points
        for idx, train_point in enumerate(X_train_np):
            if distance_metric == "euclidean":
                dist = euclidean_distance(test_point, train_point)
            elif distance_metric == "manhattan":
                dist = manhattan_distance(test_point, train_point)
            else:
                raise ValueError("Unsupported distance metric")
            distances.append((dist, y_train.iloc[idx]))
        
        # Get the k nearest neighbors
        distances.sort(key=lambda x: x[0])
        k_nearest = [label for _, label in distances[:k]]
        # Majority vote for classification
        most_common = Counter(k_nearest).most_common(1)[0][0]
        predictions.append(most_common)
    return np.array(predictions)

# ---------------------------------------------------
# 5. Evaluate k-NN for Different K Values and Distance Metrics
# ---------------------------------------------------
k_values = [1, 3, 5, 7, 9]
euclidean_accuracies = []
manhattan_accuracies = []

for k in k_values:
    # Using Euclidean distance
    y_pred_euclidean = knn_predict(X_train, y_train, X_test, k, distance_metric="euclidean")
    acc_euclidean = accuracy_score(y_test, y_pred_euclidean)
    euclidean_accuracies.append(acc_euclidean)
    
    # Using Manhattan distance
    y_pred_manhattan = knn_predict(X_train, y_train, X_test, k, distance_metric="manhattan")
    acc_manhattan = accuracy_score(y_test, y_pred_manhattan)
    manhattan_accuracies.append(acc_manhattan)
    
    print(f"K = {k}: Euclidean Accuracy = {acc_euclidean:.4f}, Manhattan Accuracy = {acc_manhattan:.4f}")

# ---------------------------------------------------
# 6. Plot Accuracy vs. K
# ---------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(k_values, euclidean_accuracies, marker='o', label='Euclidean')
plt.plot(k_values, manhattan_accuracies, marker='s', label='Manhattan')
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. K for k-NN")
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------------------------------
# 7. Confusion Matrix and Classification Report
# ---------------------------------------------------
# Choose the best k based on Euclidean accuracy
best_index = np.argmax(euclidean_accuracies)
best_k = k_values[best_index]
print(f"\nBest K based on Euclidean distance: {best_k}")

best_predictions = knn_predict(X_train, y_train, X_test, best_k, distance_metric="euclidean")

cm = confusion_matrix(y_test, best_predictions)
cr = classification_report(y_test, best_predictions)

print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(cr)
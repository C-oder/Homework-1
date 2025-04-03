import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Fetch and preprocess dataset
wine = fetch_ucirepo(id=109)
X = wine.data.features
y = wine.data.targets

# Handle missing values
X.fillna(X.mean(), inplace=True)

# Normalize data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split into Training & Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 2: Implement k-NN from scratch
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def manhattan_distance(p1, p2):
    return np.sum(np.abs(p1 - p2))

def knn_predict(X_train, y_train, X_test, k, distance_metric="euclidean"):
    predictions = []
    for test_point in X_test.to_numpy():
        distances = []
        for i, train_point in enumerate(X_train.to_numpy()):
            if distance_metric == "euclidean":
                dist = euclidean_distance(test_point, train_point)
            elif distance_metric == "manhattan":
                dist = manhattan_distance(test_point, train_point)
            distances.append((dist, y_train.iloc[i]))

        # Sort by distance and get k nearest neighbors
        distances.sort(key=lambda x: x[0])
        k_nearest = [label for _, label in distances[:k]]
        
        # Determine the most common class
        most_common = Counter(k_nearest).most_common(1)[0][0]
        predictions.append(most_common)
    
    return np.array(predictions)

# Step 3: Evaluate k-NN for different values of K
k_values = [1, 3, 5, 7, 9]
euclidean_accuracies = []
manhattan_accuracies = []

for k in k_values:
    # Euclidean
    y_pred_euclidean = knn_predict(X_train, y_train, X_test, k, "euclidean")
    acc_euclidean = accuracy_score(y_test, y_pred_euclidean)
    euclidean_accuracies.append(acc_euclidean)

    # Manhattan
    y_pred_manhattan = knn_predict(X_train, y_train, X_test, k, "manhattan")
    acc_manhattan = accuracy_score(y_test, y_pred_manhattan)
    manhattan_accuracies.append(acc_manhattan)

    print(f"\nK={k} | Euclidean Accuracy: {acc_euclidean:.4f} | Manhattan Accuracy: {acc_manhattan:.4f}")

# Step 4: Plot Accuracy vs. K
plt.figure(figsize=(8, 5))
plt.plot(k_values, euclidean_accuracies, marker='o', label='Euclidean')
plt.plot(k_values, manhattan_accuracies, marker='s', label='Manhattan')
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. K for k-NN")
plt.legend()
plt.grid()
plt.show()

# Step 5: Confusion Matrix and Classification Report (for best K)
best_k = k_values[np.argmax(euclidean_accuracies)]
best_pred = knn_predict(X_train, y_train, X_test, best_k, "euclidean")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, best_pred))

print("\nClassification Report:")
print(classification_report(y_test, best_pred))
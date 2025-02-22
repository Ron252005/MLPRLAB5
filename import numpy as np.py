import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#I am taking sample dataset for this
df = pd.DataFrame({
    'Feature1': [1, 2, 3, 6, 7, 8],
    'Feature2': [2, 3, 4, 7, 8, 9],
    'Label': [0, 0, 0, 1, 1, 1]
})

X = df.iloc[:, :-1].values  # Features
y = df.iloc[:, -1].values   # Labels

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a basic KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluate model
y_pred = knn.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

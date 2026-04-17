import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("../dataset/heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

# Controlled Tree (Overfitting fix)
dt_limited = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_limited.fit(X_train, y_train)
y_pred_limited = dt_limited.predict(X_test)

print("Controlled Tree Accuracy:", accuracy_score(y_test, y_pred_limited))

# Visualize tree
plt.figure(figsize=(15,10))
plot_tree(dt_limited, filled=True)
plt.savefig("../outputs/tree.png")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# Feature Importance
importances = rf.feature_importances_
for i, col in enumerate(X.columns):
    print(f"{col}: {importances[i]}")

# Cross Validation
scores = cross_val_score(rf, X, y, cv=5)
print("Cross-validation scores:", scores)
print("Average CV Score:", scores.mean())

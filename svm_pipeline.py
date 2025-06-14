import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                             roc_curve, auc, precision_recall_curve)
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# Configuration
config = {
    "random_seed": 42,
    "test_size": 0.3,
    "top_k_features": 5,
    "pca_components": 2,
    "shap_sample_size": 50,
    "param_grid": [
        {"kernel": ["linear"], "C": [0.1, 1, 10]},
        {"kernel": ["rbf"], "C": [0.1, 1, 10], "gamma": [0.01, 0.1, 1]},
        {"kernel": ["poly"], "C": [0.1, 1], "degree": [2, 3]},
        {"kernel": ["sigmoid"], "C": [0.1, 1], "gamma": [0.01, 0.1]}
    ]
}

# Base directory (current working directory)
base_dir = os.getcwd()

# Create folders relative to base_dir
folders = [
    os.path.join(base_dir, "visuals", "eda"),
    os.path.join(base_dir, "visuals", "metrics"),
    os.path.join(base_dir, "visuals", "shap"),
    os.path.join(base_dir, "results")
]
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Plot class distribution
sns.countplot(x=y)
plt.title("Target Class Distribution")
path = os.path.join(base_dir, "visuals", "eda", "target_distribution.png")
plt.savefig(path)
plt.close()
print(f"Saved plot: {path}")

# Top-K feature selection
selector = SelectKBest(score_func=f_classif, k=config['top_k_features'])
selector.fit(X, y)
top_features = X.columns[selector.get_support()]

# Boxplots of top features
for col in top_features:
    sns.boxplot(x=y, y=X[col])
    plt.title(f"{col} by Diagnosis")
    path = os.path.join(base_dir, "visuals", "eda", f"box_{col}.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved plot: {path}")

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.tight_layout()
path = os.path.join(base_dir, "visuals", "eda", "correlation_matrix.png")
plt.savefig(path)
plt.close()
print(f"Saved plot: {path}")

# Preprocess features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=config["test_size"], random_state=config["random_seed"]
)

# Grid Search for SVM
grid = GridSearchCV(SVC(), config["param_grid"], cv=5, scoring="accuracy")
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# Evaluate
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Best model: {grid.best_params_}")
print(f"✅ Test Accuracy: {acc:.4f}")

# Save model and metrics
report = classification_report(y_test, y_pred, output_dict=True)
report_path = os.path.join(base_dir, "results", "classification_report.csv")
pd.DataFrame(report).transpose().to_csv(report_path)
print(f"Saved classification report: {report_path}")

model_path = os.path.join(base_dir, "results", "best_model.joblib")
joblib.dump(best_model, model_path)
print(f"Saved model: {model_path}")

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
path = os.path.join(base_dir, "visuals", "metrics", "confusion_matrix.png")
plt.savefig(path)
plt.close()
print(f"Saved plot: {path}")

# ROC Curve
probs = best_model.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
path = os.path.join(base_dir, "visuals", "metrics", "roc_curve.png")
plt.savefig(path)
plt.close()
print(f"Saved plot: {path}")

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, probs)
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
path = os.path.join(base_dir, "visuals", "metrics", "precision_recall.png")
plt.savefig(path)
plt.close()
print(f"Saved plot: {path}")

# PCA Visualization of Decision Boundary
pca = PCA(n_components=config["pca_components"])
X_pca = pca.fit_transform(X_scaled)

vis_model = SVC(kernel="rbf", C=1, gamma=0.1)
vis_model.fit(X_pca, y)

x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
Z = vis_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap="coolwarm", alpha=0.3)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", edgecolors="k")
plt.title("Decision Boundary (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
path = os.path.join(base_dir, "visuals", "metrics", "pca_decision_boundary.png")
plt.savefig(path)
plt.close()
print(f"Saved plot: {path}")

# SHAP Explainability
explainer = shap.KernelExplainer(best_model.decision_function, X_train[:100])
shap_values = explainer.shap_values(X_test[:config["shap_sample_size"]], nsamples=100)

shap.summary_plot(
    shap_values,
    pd.DataFrame(X_test[:config["shap_sample_size"]], columns=X.columns),
    show=False
)
path = os.path.join(base_dir, "visuals", "shap", "shap_summary.png")
plt.savefig(path)
plt.close()
print(f"Saved plot: {path}")

print("\n🎉 All tasks completed: Model trained, evaluated, visualized, and explained using SHAP.")

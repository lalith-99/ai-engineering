"""
Day 13: Classical ML algorithms from scratch and with scikit-learn.

Covers:
1. Linear regression (from scratch + sklearn)
2. Logistic regression (sklearn)
3. Decision tree + Random Forest
4. XGBoost
5. K-Means clustering
6. Train/test split, cross-validation, metrics

Usage:
    python classical_ml.py
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, confusion_matrix, classification_report,
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("level=info event=xgboost_unavailable detail='skipping example' install='pip install xgboost'")


def section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 50}")
    print(f"event=section_start title={title!r}")
    print(f"{'=' * 50}\n")


# ========== 1. LINEAR REGRESSION FROM SCRATCH ==========

def linear_regression_scratch():
    section("Linear Regression (from scratch)")

    # Generate data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X.squeeze() + np.random.randn(100) * 0.5

    # Add bias term
    X_b = np.c_[np.ones((100, 1)), X]

    # Normal equation: theta = (X^T X)^-1 X^T y
    theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    print(f"Learned parameters: bias={theta[0]:.2f}, weight={theta[1]:.2f}")
    print(f"True parameters:    bias=4.00, weight=3.00")

    # Predictions
    y_pred = X_b @ theta
    mse = np.mean((y - y_pred) ** 2)
    print(f"MSE: {mse:.4f}")


# ========== 2. LINEAR REGRESSION WITH SKLEARN ==========

def linear_regression_sklearn():
    section("Linear Regression (sklearn)")

    X, y = make_regression(n_samples=200, n_features=3, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Coefficients: {model.coef_.round(2)}")
    print(f"Intercept: {model.intercept_:.2f}")
    print(f"R² score: {r2_score(y_test, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"CV R² scores: {cv_scores.round(3)}")
    print(f"CV mean: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")


# ========== 3. LOGISTIC REGRESSION ==========

def logistic_regression_demo():
    section("Logistic Regression")

    X, y = make_classification(n_samples=300, n_features=5, n_informative=3,
                                random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                          random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(random_state=42)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall: {recall_score(y_test, y_pred):.3f}")
    print(f"F1: {f1_score(y_test, y_pred):.3f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"\n{classification_report(y_test, y_pred)}")


# ========== 4. DECISION TREE + RANDOM FOREST ==========

def tree_and_forest():
    section("Decision Tree vs Random Forest")

    X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                                random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                          random_state=42)

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    dt_acc = accuracy_score(y_test, dt.predict(X_test))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))

    print(f"Decision Tree accuracy: {dt_acc:.3f}")
    print(f"Random Forest accuracy: {rf_acc:.3f}")
    print(f"Improvement: +{(rf_acc - dt_acc) * 100:.1f}%")

    # Feature importance (top 5)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:5]
    print(f"\nTop 5 feature importances:")
    for i, idx in enumerate(indices):
        print(f"  Feature {idx}: {importances[idx]:.3f}")

    # Cross-validation comparison
    dt_cv = cross_val_score(dt, X, y, cv=5).mean()
    rf_cv = cross_val_score(rf, X, y, cv=5).mean()
    print(f"\nCV scores — DT: {dt_cv:.3f}, RF: {rf_cv:.3f}")


# ========== 5. XGBOOST ==========

def xgboost_demo():
    if not HAS_XGBOOST:
        return

    section("XGBoost")

    X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                                random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                          random_state=42)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"F1: {f1_score(y_test, y_pred):.3f}")

    cv = cross_val_score(model, X, y, cv=5).mean()
    print(f"CV accuracy: {cv:.3f}")


# ========== 6. K-MEANS CLUSTERING ==========

def kmeans_demo():
    section("K-Means Clustering")

    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8,
                            random_state=42)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X)

    print(f"Cluster centers:\n{kmeans.cluster_centers_.round(2)}")
    print(f"Inertia (within-cluster sum of squares): {kmeans.inertia_:.1f}")

    # Elbow method — try different k values
    print(f"\nElbow method:")
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        print(f"  k={k}: inertia={km.inertia_:.1f}")


if __name__ == "__main__":
    linear_regression_scratch()
    linear_regression_sklearn()
    logistic_regression_demo()
    tree_and_forest()
    xgboost_demo()
    kmeans_demo()

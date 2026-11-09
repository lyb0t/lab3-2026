import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
    ClassificationMetrics
)

# --- Вспомогательные функции ---
def sigmoid(z):
    """Функция активации сигмоида"""
    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

def compute_mse(y_true, y_pred):
    """Среднеквадратичная ошибка"""
    return np.mean((y_true - y_pred) ** 2)

def compute_log_loss(y_true, y_pred_proba):
    """Бинарная кросс-энтропия"""
    epsilon = 1e-15
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))


# --- 1. Линейная регрессия ---
def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""
    X_train = np.array(data.train_features, dtype=float)
    y_train = np.array(data.train_target, dtype=float)
    X_test = np.array(data.test_features, dtype=float)
    lr = data.learning_rate
    epochs = data.epochs
    
    n_samples, n_features = X_train.shape
    weights = np.zeros(n_features)
    bias = 0.0
    loss_history = []
    
    for epoch in range(epochs):
        y_pred = np.dot(X_train, weights) + bias
        loss = compute_mse(y_train, y_pred)
        loss_history.append(float(loss))
        
        dw = (1 / n_samples) * np.dot(X_train.T, (y_pred - y_train))
        db = (1 / n_samples) * np.sum(y_pred - y_train)
        
        weights -= lr * dw
        bias -= lr * db
        
    test_pred = np.dot(X_test, weights) + bias
    
    return RegressionResult(
        weights=weights.tolist(),
        bias=float(bias),
        loss_history=loss_history,
        predictions=test_pred.tolist()
    )


# --- 2. Логистическая регрессия ---
def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
    X_train = np.array(data.train_features, dtype=float)
    y_train = np.array(data.train_target, dtype=float)
    X_test = np.array(data.test_features, dtype=float)
    lr = data.learning_rate
    epochs = data.epochs
    
    n_samples, n_features = X_train.shape
    weights = np.zeros(n_features)
    bias = 0.0
    loss_history = []
    
    for epoch in range(epochs):
        z = np.dot(X_train, weights) + bias
        y_pred_proba = sigmoid(z)
        
        loss = compute_log_loss(y_train, y_pred_proba)
        loss_history.append(float(loss))
        
        dw = (1 / n_samples) * np.dot(X_train.T, (y_pred_proba - y_train))
        db = (1 / n_samples) * np.sum(y_pred_proba - y_train)
        
        weights -= lr * dw
        bias -= lr * db
        
    test_z = np.dot(X_test, weights) + bias
    test_proba = sigmoid(test_z)
    test_preds_binary = (test_proba >= 0.5).astype(int)
    
    return ClassificationResult(
        weights=weights.tolist(),
        bias=float(bias),
        loss_history=loss_history,
        probabilities=test_proba.tolist(),
        predictions=test_preds_binary.tolist()
    )


# --- 3. Сравнение моделей ---
def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split."""
    X_train = np.array(data.train_features, dtype=float)
    y_train = np.array(data.train_target, dtype=float)
    X_test = np.array(data.test_features, dtype=float)
    y_test = np.array(data.test_target, dtype=float)
    
    def get_metrics(y_true, y_pred, y_proba):
        return ClassificationMetrics(
            roc_auc=float(roc_auc_score(y_true, y_proba)),
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, zero_division=0))
        )

    # Своя логистическая регрессия
    my_log_reg_res = train_logistic_classifier(data)
    my_metrics = get_metrics(y_test, my_log_reg_res.predictions, my_log_reg_res.probabilities)

    # Logistic Regression (Sklearn)
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_proba = lr_model.predict_proba(X_test)[:, 1]
    lr_metrics = get_metrics(y_test, lr_pred, lr_proba)

    # Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    dt_proba = dt_model.predict_proba(X_test)[:, 1]
    dt_metrics = get_metrics(y_test, dt_pred, dt_proba)

    # KNN
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train)
    knn_pred = knn_model.predict(X_test)
    knn_proba = knn_model.predict_proba(X_test)[:, 1]
    knn_metrics = get_metrics(y_test, knn_pred, knn_proba)

    return BaselineComparisonResult(
        logistic_regression=my_metrics,
        decision_tree=dt_metrics,
        knn=knn_metrics
    )
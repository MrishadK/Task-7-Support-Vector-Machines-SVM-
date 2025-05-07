# Task 7 - Support Vector Machines (SVM)

# 1. Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# 2. Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# 3. Preprocessing - Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Train SVM with Linear Kernel
svm_linear = SVC(kernel='linear', C=1, random_state=42)
svm_linear.fit(X_train, y_train)
linear_score = svm_linear.score(X_test, y_test)
print(f"Linear Kernel Test Accuracy: {linear_score:.4f}")

# 6. Train SVM with RBF Kernel
svm_rbf = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
svm_rbf.fit(X_train, y_train)
rbf_score = svm_rbf.score(X_test, y_test)
print(f"RBF Kernel Test Accuracy: {rbf_score:.4f}")

# 7. Visualization using PCA (2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Retrain on 2D data
svm_rbf_2d = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
svm_rbf_2d.fit(X_pca, y)

def plot_decision_boundary(clf, X, y, title):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.show()

# Plot
plot_decision_boundary(svm_rbf_2d, X_pca, y, "SVM with RBF Kernel (PCA Reduced Data)")

# 8. Hyperparameter Tuning using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.01, 0.1, 1]
}

grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
grid.fit(X_scaled, y)

print("Best Parameters from Grid Search:", grid.best_params_)

# 9. Cross-validation with best parameters
best_model = SVC(kernel='rbf', C=grid.best_params_['C'], gamma=grid.best_params_['gamma'])
cv_scores = cross_val_score(best_model, X_scaled, y, cv=5)
print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f}")

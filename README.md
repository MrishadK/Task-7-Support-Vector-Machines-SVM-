# Task 7 - Support Vector Machines (SVM)


## 🔥 What I Did
- Loaded the **Breast Cancer Dataset** from scikit-learn.
- Scaled features using **StandardScaler**.
- Trained **SVM** models with:
  - **Linear Kernel**
  - **RBF (Radial Basis Function) Kernel**
- Visualized the decision boundary after reducing features to 2D using **PCA**.
- Tuned hyperparameters (`C` and `gamma`) using **GridSearchCV**.
- Evaluated model using **5-fold Cross-Validation**.

---

## 🛠 Libraries Used
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
---

## 📊 Results
| Model              | Test Accuracy |
|--------------------|---------------|
| SVM (Linear Kernel) | ~0.9561      |
| SVM (RBF Kernel)    | ~0.9737        |
| Best Cross-Validation Accuracy (after tuning) | ~0.9789 |

![image](https://github.com/user-attachments/assets/bf9cf791-11a6-4373-bc5c-fe5182b127cb)

---

## 📷 Visualization
> SVM Decision Boundary plotted on 2D data (reduced using PCA).

![Decision Boundary Placeholder](https://via.placeholder.com/600x300?text=Decision+Boundary+Plot)

![image](https://github.com/user-attachments/assets/aa10efc5-0608-4503-8d16-038fdf3462c5)

---

## 🔗 Dataset Source
- [Breast Cancer Dataset - Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

---

# 🧠 MNIST Digit Classification – ML Models Comparison + SVM Hyperparameter Tuning

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)

## 📌 Project Overview

This project implements a **complete Handwritten Digit Classification system** using the **MNIST dataset**. I trained and compared **9 different machine learning models**, evaluated them using **ROC-AUC curves**, selected the best performer (SVM), and fine-tuned it using **GridSearchCV** for optimal performance.

---

## 🎯 Project Objectives

 ▫️Compare multiple ML algorithms systematically  
 ▪️Visualize model performance using ROC curves  
 ▫️Optimize the best model through hyperparameter tuning  
 ▫️Deploy a production-ready classifier  

---

## Click here for Dataset 📚📌
<div align="side"> 
    
[![Kaggle](https://img.shields.io/badge/Kaggle-100000?style=for-the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
</div>

## 🚀 Project Workflow

### 1️⃣. Data Loading & Preprocessing
- Loaded MNIST dataset in CSV format (`label + 784 pixel features`)
- Handled missing values and data inconsistencies
- Normalized pixel values (0-255 → 0-1)
- Split data into **training** and **testing** sets

### 2️⃣. Model Training & Implementation
Built a custom Python class `MNISTModel` that trains **9 ML algorithms**:

| Model | Algorithm Type |
|-------|---------------|
| 🔹 K-Nearest Neighbors (KNN) | Instance-based |
| 🔹 Naive Bayes | Probabilistic |
| 🔹 Logistic Regression | Linear classifier |
| 🔹 Decision Tree | Tree-based |
| 🔹 Random Forest | Ensemble (Bagging) |
| 🔹 AdaBoost | Ensemble (Boosting) |
| 🔹 Gradient Boosting | Ensemble (Boosting) |
| 🔹 XGBoost | Optimized Gradient Boosting |
| 🔹 Support Vector Machine (SVM) | Kernel-based |

**Each model:**
- Prints training accuracy
- Stores trained model for comparison
- Supports probability predictions for ROC analysis

### 3️⃣. Model Comparison Using ROC Curves
- Computed `predict_proba` for all models
- Converted **multiclass problem → binary** (One-vs-Rest for Class 1)
- Plotted **all 9 ROC curves** on a single graph
- Calculated **AUC (Area Under Curve)** for each model
- **Result:** SVM with RBF kernel achieved the highest AUC → selected as final model

### 4️⃣. Hyperparameter Tuning with GridSearchCV
Fine-tuned the SVM model by testing combinations of:
```python
param_grid = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01],
    'decision_function_shape': ['ovr', 'ovo'],
    'probability': [True]
}
```

- Used **cross-validation**
- Identified optimal hyperparameters
- Significantly improved model stability and accuracy

### 5️⃣. Final Model Evaluation
Trained the final SVM with best parameters and evaluated using:

📊 **Metrics:**
- Training Accuracy
- Testing Accuracy
- **Confusion Matrix** (10×10 for digits 0-9)
- **Classification Report** (Precision, Recall, F1-Score per digit)

### 6️⃣. Model Deployment
- Saved the final trained SVM using `pickle`
- Tested predictions on random 784-pixel inputs
- Model ready for production deployment

---

## 🔍 Key Insights & Learnings

💡 **ROC-AUC provides better model comparison** than simple accuracy, especially for imbalanced datasets  
💡 **SVM with RBF kernel consistently outperformed** other traditional ML models on MNIST  
💡 **GridSearchCV tuning significantly improved** model generalization and reduced overfitting  
💡 **Object-Oriented Programming (OOP)** approach with classes made code modular and maintainable  
💡 **Exception handling** ensured robust, production-ready code  

---

## 📂 Project Structure
```
📦 MNIST-Digit-Classification
├── 📄 mnist_classification.ipynb    # Main Jupyter Notebook
├── 📄 mnist_model.py                # MNISTModel class implementation
├── 📁 data/
│   ├── mnist_train.csv
│   └── mnist_test.csv
├── 📁 models/
│   └── best_svm_model.pkl          # Saved final model
├── 📁 visualizations/
│   ├── roc_curves.png
│   └── confusion_matrix.png
└── 📄 README.md
```

---

## 🛠️ Technologies & Libraries Used

**Languages:**
- Python 3.x

**Libraries:**
```python
# Data Manipulation
import numpy as np
import pandas as pd

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score

# ML Algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Model Persistence
import pickle
```

---

## 📊 Results Summary

| Model | Test Accuracy | ROC-AUC (Class 1) |
|-------|--------------|-------------------|
| KNN | ~96.5% | 0.985 |
| Naive Bayes | ~84.2% | 0.921 |
| Logistic Regression | ~92.3% | 0.968 |
| Decision Tree | ~87.6% | 0.894 |
| Random Forest | ~96.8% | 0.992 |
| AdaBoost | ~79.4% | 0.912 |
| Gradient Boosting | ~95.7% | 0.989 |
| XGBoost | ~96.9% | 0.993 |
| **SVM (RBF)** | **97.8%** | **0.997** ✅ |
| **SVM (Tuned)** | **98.2%** | **0.998** 🏆 |

*Note: Replace with your actual results*

---

## 🚀 How to Run This Project

### Prerequisites
```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
```

### Steps
1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/MNIST-Digit-Classification.git
cd MNIST-Digit-Classification
```

2. **Run the Jupyter Notebook:**
```bash
jupyter notebook mnist_classification.ipynb
```

3. **Or run the Python script:**
```bash
python mnist_model.py
```

4. **Load the saved model for predictions:**
```python
import pickle

# Load model
with open('models/best_svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make prediction
prediction = model.predict([your_784_pixel_array])
print(f"Predicted Digit: {prediction[0]}")
```

---

## 📈 Sample Visualizations

### ROC Curves Comparison
![ROC Curves](Roc-Auc_curve.jpg)
*All 9 models plotted together showing SVM's superior performance*

---

## 🎓 Learning Outcomes

Through this project, I gained hands-on experience with:

✅ **Multi-model ML pipeline development**  
✅ **ROC curve analysis and interpretation**  
✅ **Hyperparameter optimization techniques**  
✅ **Object-oriented ML code design**  
✅ **Model evaluation best practices**  
✅ **Production-ready model deployment**  

---

## 🔮 Future Enhancements

- [ ] Implement **Deep Learning** models (CNN) for comparison
- [ ] Create a **web app** using Streamlit/Flask for digit drawing and prediction
- [ ] Add **model explainability** using LIME/SHAP
- [ ] Optimize for **real-time inference** speed
- [ ] Deploy on **cloud platform** (AWS/GCP/Azure)

---

## 🙏 Acknowledgements

First & most a very Special thanks to [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sai-kamal-korlakunta/), Trainer & CEO at **Vihara Tech** 👉 [viharatech.com](https://www.viharatech.com/)  
For his invaluable guidance throughout my **Data Science with Gen AI** journey.

---

## 🌐 Connect With Me

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/karthik-vana)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/karthik-vana)
[![HackerRank](https://img.shields.io/badge/HackerRank-2EC866?style=for-the-badge&logo=hackerrank&logoColor=white)](https://www.hackerrank.com/profile/karthikvana236)
[![Portfolio](https://img.shields.io/badge/Portfolio-FF5722?style=for-the-badge&logo=google-chrome&logoColor=white)](https://your-portfolio-link.com)

</div>

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star! ⭐**

*Built with ❤️ & python🐍 as part of my Data Science journey*

</div>

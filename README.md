# Loan Approval Prediction

## 📌 Overview
Loan approval is a pivotal process for financial institutions, balancing customer satisfaction with risk management. This project employs machine learning techniques to predict loan approvals using applicant profiles and financial attributes. By leveraging data-driven insights, the model optimizes loan approval processes and minimizes default risks.

---

## 📊 Dataset
- **Source**: [Kaggle - Loan Approval Prediction](https://kaggle.com/competitions/playground-series-s4e10)
- **Size**: 58,645 records, 13 features
- **Key Attributes**:
  - **Applicant Information**: Age, income, employment length, home ownership.
  - **Loan Details**: Amount, interest rate, grade.
  - **Credit History**: Length and default status.

---

## 🎯 Project Goals
1. Build a reliable loan approval prediction model with at least **90% accuracy**.
2. Identify key factors influencing loan decisions.
3. Provide actionable insights for financial institutions.

---

## 🔑 Key Features
### 1. Exploratory Data Analysis (EDA)
- Visualizations to explore patterns and relationships.
- Correlation heatmaps for feature insights.
- Handling imbalanced data.

### 2. Feature Engineering
- Derived metrics such as:
  - **Debt-to-Income Ratio**
  - **Credit History-to-Age Ratio**

### 3. Machine Learning Models
- **Algorithms**:
  - Logistic Regression, KNN, Decision Tree, Random Forest, XGBoost
- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1-score, ROC-AUC
- **Best Model**:
  - XGBoost with **95.2% accuracy** and **95.1% ROC-AUC**.

---

## 🛠️ Implementation
### Tools & Libraries
- **Data Analysis & Visualization**: `numpy`, `pandas`, `seaborn`, `matplotlib`
- **Machine Learning**: `scikit-learn`, `xgboost`

### Steps
1. **Data Preprocessing**:
   - Handled missing and duplicate values.
   - Categorical encoding and scaling of numerical features.
2. **Feature Engineering**:
   - Introduced new attributes for better prediction.
3. **Model Training**:
   - Used GridSearchCV for hyperparameter tuning.
4. **Evaluation**:
   - Validated models with K-Fold Cross-Validation.

---

## 🚀 Results
- **Best Model**: XGBoost
  - **Accuracy**: 95.2%
  - **ROC-AUC**: 95.1%
- **Key Insights**:
  - Longer credit history and higher income positively influence approvals.
  - Default history and high debt-to-income ratios reduce approval likelihood.

---

## ⚙️ How to Run
### Prerequisites
- Python 3.7+
- Required Libraries: Install via `requirements.txt`

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>

### 📈 Future Improvements
- Experiment with advanced techniques like deep learning.
- Address class imbalance with SMOTE or similar methods.
- Explore additional datasets for better generalization.

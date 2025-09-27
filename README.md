# ❤️ Heart Disease Prediction Project

## 📌 Overview
This project uses the **UCI Heart Disease dataset** to predict the likelihood of heart disease in patients.  
The pipeline covers **data preprocessing, feature selection, dimensionality reduction, supervised & unsupervised learning, hyperparameter tuning, and deployment**.  

---

## 🗂 Workflow

### **2.1 Data Preprocessing & Cleaning**
**Steps**
1. Load the Heart Disease UCI dataset into a Pandas DataFrame.  
2. Handle missing values (imputation or removal).  
3. Perform data encoding (one-hot encoding for categorical variables).  
4. Standardize numerical features using **MinMaxScaler** or **StandardScaler**.  
5. Conduct **Exploratory Data Analysis (EDA)**:
   - Histograms
   - Correlation heatmaps
   - Boxplots  

**Deliverable**
✔ Cleaned dataset ready for modeling  

---

### **2.2 Dimensionality Reduction - PCA**
**Steps**
1. Apply **Principal Component Analysis (PCA)** to reduce dimensionality while retaining variance.  
2. Determine optimal number of components using the explained variance ratio.  
3. Visualize results:
   - PCA scatter plot
   - Cumulative variance plot  

**Deliverable**
✔ PCA-transformed dataset  
✔ Variance explained graph  

---

### **2.3 Feature Selection**
**Steps**
1. Use **Feature Importance** (Random Forest / XGBoost) to rank features.  
2. Apply **Recursive Feature Elimination (RFE)** to select predictors.  
3. Use **Chi-Square Test** for feature significance.  
4. Retain only most relevant features.  

**Deliverable**
✔ Reduced dataset with selected features  
✔ Feature importance ranking visualization  

---

### **2.4 Supervised Learning - Classification Models**
**Steps**
1. Split dataset into **train (80%)** and **test (20%)**.  
2. Train classification models:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Support Vector Machine (SVM)  
3. Evaluate using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC Curve & AUC  

**Deliverable**
✔ Trained models with evaluation metrics  

---

### **2.5 Unsupervised Learning - Clustering**
**Steps**
1. Apply **K-Means Clustering** (elbow method to find K).  
2. Perform **Hierarchical Clustering** (dendrogram analysis).  
3. Compare clusters with actual disease labels.  

**Deliverable**
✔ Clustering models with visualizations  

---

### **2.6 Hyperparameter Tuning**
**Steps**
1. Apply **GridSearchCV** and **RandomizedSearchCV** for optimization.  
2. Compare tuned models with baseline performance.  

**Deliverable**
✔ Best-performing optimized model  

---

### **2.7 Model Export & Deployment**
**Steps**
1. Save trained model with **joblib** or **pickle** (`.pkl` format).  
2. Store preprocessing pipeline + model for reproducibility.  

**Deliverable**
✔ Exported `.pkl` model  

---

### **2.8 Streamlit Web UI Development**
**Steps**
1. Build **Streamlit UI** for user health data input.  
2. Display real-time prediction outputs.  
3. Add interactive data visualizations.  

**Deliverable**
✔ Functional Streamlit app for predictions & EDA  

---

## ⚙️ Requirements
- Python 3.8+  
- Libraries:
  - pandas  
  - numpy  
  - scikit-learn  
  - matplotlib  
  - seaborn  
  - xgboost  
  - joblib / pickle  
  - streamlit (for UI)  

Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost joblib streamlit

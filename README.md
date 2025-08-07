# Heart Disease Prediction - Machine Learning Project

## 📋 Project Overview

This project implements a comprehensive machine learning analysis for heart disease prediction using various classification algorithms. The goal is to predict heart disease outcomes based on clinical and laboratory parameters.

**Developer:** Zayyan  
**Project Type:** Machine Learning Classification  
**Target Variable:** Heart Disease Outcome (Binary Classification)

## 🎯 Project Objectives

- Perform exploratory data analysis on heart disease dataset
- Implement and compare multiple machine learning algorithms
- Identify the most important features for heart disease prediction
- Evaluate model performance using accuracy and AUC metrics
- Determine the best performing model for clinical application

## 📊 Dataset Information

**Dataset:** Heart Disease.csv  
**Source:** Provided by university instructor for academic purposes  
**Size:** 1,177 patients with 51 features  
**Target Variable:** `outcome (Target)` - Binary classification (0: No heart disease, 1: Heart disease)
**Note:** This dataset is not owned by the developer and was provided by the university teacher for this academic project.

### Key Features Analyzed:
- **Demographics:** Age, Gender, BMI
- **Medical Conditions:** Hypertension, Diabetes, Depression, COPD, etc.
- **Vital Signs:** Heart rate, Blood pressure, Temperature, SP O2
- **Laboratory Values:** Blood counts, Electrolytes, Enzymes, etc.

## 🛠️ Technologies Used

- **Python 3.x**
- **Libraries:**
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computations
  - `matplotlib` & `seaborn` - Data visualization
  - `scikit-learn` - Machine learning algorithms
  - `GridSearchCV` - Hyperparameter tuning

## 📓 Notebook Structure (`code.ipynb`)

The analysis is organized into logical sections within the Jupyter notebook:

### **Section 1: Setup & Data Loading**
- **Cell 0:** Project title and developer information
- **Cell 1-2:** Library imports and data loading
- **Cell 4:** Initial data exploration and statistics

### **Section 2: Exploratory Data Analysis**
- **Cell 5-6:** Data visualization and feature distribution analysis
- **Cell 7-8:** Correlation analysis and heatmap generation

### **Section 3: Feature Engineering**
- **Cell 9-10:** Feature selection based on correlation with target variable

### **Section 4: Model Implementation**
- **Cell 11-12:** Logistic Regression with hyperparameter tuning
- **Cell 13-14:** Naive Bayes Classifier implementation
- **Cell 15-16:** Decision Tree with feature importance analysis
- **Cell 17-18:** Random Forest with comprehensive parameter optimization
- **Cell 19-20:** K-Nearest Neighbors classification
- **Cell 21-22:** Support Vector Machine with polynomial kernel

### **Section 5: Results & Comparison**
- **Cell 23-24:** Comprehensive model comparison and visualization
- **Cell 25-26:** Final analysis and conclusions

## 🔬 Methodology

### 1. Data Preprocessing
- **Feature Selection:** Selected 9 most correlated features (>0.125 correlation with target)
- **Missing Value Handling:** Used SimpleImputer with mean strategy
- **Data Scaling:** Applied StandardScaler for normalization
- **Train/Validation/Test Split:** 60%/20%/20% split

### 2. Feature Engineering
**Selected Features:**
- Heart rate
- RDW (Red Cell Distribution Width)
- Leucocyte count
- PT (Prothrombin Time)
- INR (International Normalized Ratio)
- Urea nitrogen
- Blood potassium
- Anion gap
- Lactic acid

### 3. Model Implementation

Six classification algorithms were implemented and compared:

| Algorithm | Best Parameters | CV Score |
|-----------|----------------|----------|
| **Logistic Regression** | C=0.1, solver='lbfgs', max_iter=100 | 84.78% |
| **Naive Bayes** | var_smoothing=1e-09 | 80.50% |
| **Decision Tree** | criterion='entropy', max_depth=5 | 84.79% |
| **Random Forest** | n_estimators=50, max_depth=5 | 85.57% |
| **K-Nearest Neighbors** | n_neighbors=9, metric='euclidean' | 84.39% |
| **Support Vector Machine** | C=0.1, kernel='poly', gamma='scale' | 85.17% |

## 📈 Results & Performance

### Model Performance Comparison

| Model | Test Accuracy | Test AUC | Validation Accuracy | Validation AUC |
|-------|---------------|----------|-------------------|----------------|
| **Logistic Regression** | **86.05%** | **84.51%** | 89.53% | 80.13% |
| Naive Bayes | 83.72% | 79.14% | 86.05% | 82.89% |
| Decision Tree | 77.91% | 42.36% | 86.05% | 61.78% |
| Random Forest | 80.23% | 80.82% | 87.21% | 78.82% |
| KNN | 82.56% | 64.70% | 88.37% | 74.08% |
| SVM | 84.88% | 81.24% | 90.70% | 81.97% |

### 🏆 Best Performing Model

**Logistic Regression** emerged as the best model with:
- **Test Accuracy:** 86.05%
- **Test AUC:** 84.51%
- **Validation Accuracy:** 89.53%

### Feature Importance Analysis

**Top 5 Most Important Features:**
1. **Anion gap** (24.45%) - Most critical predictor
2. **Urea nitrogen** (11.62%) - Kidney function indicator
3. **Leucocyte** (11.24%) - White blood cell count
4. **INR** (11.15%) - Blood clotting indicator
5. **PT** (9.66%) - Prothrombin time

## 🔍 Key Findings

### 1. Model Performance
- **Logistic Regression** showed the best balance of accuracy and AUC
- **Decision Tree** suffered from overfitting (high train accuracy, low test AUC)
- **Random Forest** provided good feature importance insights

### 2. Clinical Insights
- **Anion gap** is the strongest predictor of heart disease
- **Blood chemistry markers** (urea, leucocyte, INR) are highly predictive
- **Vital signs** (heart rate) contribute moderately to prediction

### 3. Overfitting Analysis
- **Decision Tree** showed significant overfitting (94.26% train AUC vs 42.36% test AUC)
- **Logistic Regression** demonstrated the most stable performance across datasets

## 🚀 Usage Instructions

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Running the Analysis
1. Ensure `Heart Disease.csv` is in the project directory
2. Open `code.ipynb` in Jupyter Notebook or JupyterLab
3. Run all cells sequentially for complete analysis

### Key Code Sections
- **Data Loading & EDA:** Cells 1-8
- **Feature Selection:** Cell 10
- **Model Training:** Cells 12-22
- **Results Comparison:** Cell 24

## 📊 Visualizations Generated

The analysis produces several visualizations:
- **Feature Distribution Histograms** - Understanding data patterns
- **Correlation Heatmap** - Feature relationships
- **Target Distribution** - Class balance analysis
- **Model Performance Comparisons** - Accuracy and AUC comparisons
- **Overfitting Analysis** - Train vs Test performance gaps

## 🔬 Clinical Applications

This model can be used for:
- **Early heart disease screening** in clinical settings
- **Risk stratification** of patients
- **Resource allocation** in healthcare systems
- **Clinical decision support** tools

## 📝 Limitations

1. **Dataset Size:** Limited to 1,177 patients
2. **Feature Selection:** Only 9 features used from 51 available
3. **Cross-validation:** 5-fold CV used, could benefit from more folds
4. **External Validation:** No external dataset validation

## 🔮 Future Improvements

1. **Feature Engineering:** Create interaction terms and polynomial features
2. **Ensemble Methods:** Combine multiple models for better performance
3. **Deep Learning:** Implement neural networks for complex patterns
4. **External Validation:** Test on independent datasets
5. **Clinical Validation:** Real-world clinical trial validation

## 📞 Contact

**Developer:** Zayyan  
**Project:** Heart Disease Prediction ML Analysis

---

*This project demonstrates comprehensive machine learning workflow from data exploration to model deployment for clinical applications.*

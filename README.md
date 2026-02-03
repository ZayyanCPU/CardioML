<div align="center">

# 🫀 CardioML

### Predictive Heart Disease Classification Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![scikit learn](https://img.shields.io/badge/scikit_learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-11557c?style=for-the-badge)](https://matplotlib.org)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.x-9ADCFF?style=for-the-badge)](https://seaborn.pydata.org)

*A comprehensive machine learning analysis comparing six classification algorithms for heart disease prediction, achieving **86.05% test accuracy** with Logistic Regression.*

[Key Features](#-key-features) • [Results](#-results--insights) • [Tech Stack](#-tech-stack) • [Getting Started](#-getting-started) • [Skills](#-skills-demonstrated)

---

</div>

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Analysis Pipeline](#-analysis-pipeline)
- [Results & Insights](#-results--insights)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Skills Demonstrated](#-skills-demonstrated)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

**CardioML** is a machine learning project focused on predicting heart disease outcomes using clinical and laboratory parameters. This project demonstrates a complete data science workflow from exploratory data analysis through model deployment, comparing multiple classification algorithms to identify the optimal predictive model.

| Attribute | Details |
|-----------|---------|
| **Developer** | Zayyan |
| **Dataset** | 1,177 patients with 51 clinical features |
| **Target** | Binary classification (Heart Disease: Yes/No) |
| **Best Model** | Logistic Regression (86.05% accuracy) |
| **Techniques** | GridSearchCV, Cross Validation, Feature Engineering |

---

## ✨ Key Features

| Feature | Description |
|:-------:|-------------|
| 🔬 | **Comprehensive EDA** — In depth exploratory analysis with 48 feature distributions and correlation heatmaps |
| 🎯 | **Smart Feature Selection** — Automated selection of 9 high correlation features (>0.125) from 51 available |
| 🤖 | **Six ML Models** — Logistic Regression, Naive Bayes, Decision Tree, Random Forest, KNN, and SVM |
| ⚙️ | **Hyperparameter Tuning** — Exhaustive GridSearchCV optimization for each algorithm |
| 📊 | **Performance Visualization** — Comparative charts for accuracy, AUC, and overfitting analysis |
| 🏥 | **Clinical Relevance** — Feature importance analysis identifying key biomarkers |

---

## 🛠 Tech Stack

<div align="center">

| Category | Technologies |
|:--------:|-------------|
| **Language** | Python 3.x |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | scikit learn (Logistic Regression, Naive Bayes, Decision Tree, Random Forest, KNN, SVM) |
| **Model Optimization** | GridSearchCV, Cross Validation |
| **Environment** | Jupyter Notebook |

</div>

---

## 🔄 Analysis Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           CardioML Analysis Pipeline                            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  📥 DATA LOADING                                                                │
│  └── Load Heart Disease.csv (1,177 patients × 51 features)                      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  🔍 EXPLORATORY DATA ANALYSIS                                                   │
│  ├── Feature distribution histograms                                            │
│  ├── Target variable analysis                                                   │
│  └── Correlation matrix heatmap                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  ⚙️ PREPROCESSING                                                               │
│  ├── SimpleImputer (mean strategy)                                              │
│  ├── StandardScaler normalization                                               │
│  └── Train/Validation/Test split (60%/20%/20%)                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  🎯 FEATURE SELECTION                                                           │
│  └── 9 features selected (correlation > 0.125 with target)                      │
│      Heart rate, RDW, Leucocyte, PT, INR, Urea nitrogen,                        │
│      Blood potassium, Anion gap, Lactic acid                                    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  🤖 MODEL TRAINING (with GridSearchCV)                                          │
│  ├── Logistic Regression                                                        │
│  ├── Naive Bayes                                                                │
│  ├── Decision Tree                                                              │
│  ├── Random Forest                                                              │
│  ├── K Nearest Neighbors                                                        │
│  └── Support Vector Machine                                                     │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  📈 EVALUATION & COMPARISON                                                     │
│  ├── Accuracy & AUC metrics                                                     │
│  ├── Overfitting analysis                                                       │
│  └── Best model selection                                                       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📈 Results & Insights

### Model Performance Comparison

| Model | Test Accuracy | Test AUC | Validation Accuracy | Validation AUC | CV Score |
|:------|:-------------:|:--------:|:-------------------:|:--------------:|:--------:|
| **🏆 Logistic Regression** | **86.05%** | **84.51%** | 89.53% | 80.13% | 84.78% |
| SVM | 84.88% | 81.24% | 90.70% | 81.97% | 85.17% |
| Naive Bayes | 83.72% | 79.14% | 86.05% | 82.89% | 80.50% |
| KNN | 82.56% | 64.70% | 88.37% | 74.08% | 84.39% |
| Random Forest | 80.23% | 80.82% | 87.21% | 78.82% | 85.57% |
| Decision Tree | 77.91% | 42.36% | 86.05% | 61.78% | 84.79% |

### 🏆 Best Model: Logistic Regression

```
╔═══════════════════════════════════════════════════════════════╗
║                    WINNING MODEL METRICS                       ║
╠═══════════════════════════════════════════════════════════════╣
║  Model:               Logistic Regression                      ║
║  Best Parameters:     C=0.1, solver='lbfgs', max_iter=100      ║
║  ─────────────────────────────────────────────────────────────║
║  Test Accuracy:       86.05%                                   ║
║  Test AUC:            84.51%                                   ║
║  Validation Accuracy: 89.53%                                   ║
║  CV Score:            84.78%                                   ║
╚═══════════════════════════════════════════════════════════════╝
```

### Feature Importance Analysis

Top predictive biomarkers identified through Decision Tree analysis:

| Rank | Feature | Importance | Clinical Significance |
|:----:|---------|:----------:|----------------------|
| 1 | **Anion Gap** | 24.45% | Metabolic status indicator |
| 2 | **Urea Nitrogen** | 11.62% | Kidney function marker |
| 3 | **Leucocyte** | 11.24% | White blood cell count |
| 4 | **INR** | 11.15% | Blood clotting indicator |
| 5 | **PT** | 9.66% | Prothrombin time |

### Key Findings

- ✅ **Logistic Regression** demonstrated the best balance of accuracy and generalization
- ⚠️ **Decision Tree** showed significant overfitting (94.26% train AUC vs 42.36% test AUC)
- 📊 **Anion gap** emerged as the strongest predictor of heart disease
- 🔬 Blood chemistry markers (urea, leucocyte, INR) are highly predictive
- 💓 Vital signs contribute moderately to prediction accuracy

### Overfitting Analysis

```
Model Performance Gap (Train AUC − Test AUC):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Decision Tree:      ████████████████████████████████████████  51.90%  ⚠️ OVERFITTING
KNN:                ██████████████████████                    22.70%
Random Forest:      ████████████████                          16.20%
Logistic Regression:████████                                   8.70%  ✅ STABLE
Naive Bayes:        ██████                                     6.50%
SVM:                ██████                                     6.20%
```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.x installed, then install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ZayyanCPU/CardioML.git
   cd CardioML
   ```

2. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook CardioML.ipynb
   ```

3. **Run all cells** sequentially for complete analysis

### Quick Start

```python
# Load and explore the data
import pandas as pd
data = pd.read_csv('Heart Disease.csv')
print(f"Dataset: {data.shape[0]} patients, {data.shape[1]} features")

# Run the notebook for full analysis
```

---

## 📁 Project Structure

```
CardioML/
│
├── 📓 CardioML.ipynb        # Main analysis notebook
├── 📊 Heart Disease.csv     # Dataset (1,177 patients)
├── 📖 README.md             # Project documentation
└── 📄 LICENSE               # MIT License
```

---

## 💼 Skills Demonstrated

| Category | Skills |
|----------|--------|
| **Data Science** | Exploratory Data Analysis, Feature Engineering, Statistical Analysis |
| **Machine Learning** | Classification Algorithms, Hyperparameter Tuning, Model Evaluation |
| **Python** | Pandas, NumPy, scikit learn, Matplotlib, Seaborn |
| **Best Practices** | Cross Validation, Train/Val/Test Splits, Overfitting Detection |
| **Domain Knowledge** | Healthcare Analytics, Clinical Biomarker Interpretation |
| **Documentation** | Technical Writing, Data Visualization, Results Presentation |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

### 📬 Connect

**Developed by Zayyan**

*Building intelligent healthcare solutions through data science*

⭐ Star this repository if you found it helpful!

---

</div>

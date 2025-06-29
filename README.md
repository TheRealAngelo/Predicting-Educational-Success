<h1 align="center"> Comparative Analysis of Machine Learning Models for Predicting Educational Success Using Study Patterns </h1>

<h2 align="center"> This project analyzes and compares multiple machine learning models to predict student final grades using a large educational dataset. The workflow is implemented in a Jupyter notebook and includes data preprocessing, model training, evaluation, and visualization.</h2>

---

## Features

- **Data Preprocessing:** Cleans and encodes categorical variables, handles missing values, and prepares features/target.
- **Model Comparison:** Evaluates 8 classification models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Naive Bayes
  - k-Nearest Neighbors
  - Support Vector Machine
  - Neural Network (MLP)
  - XGBoost (with GPU/CPU support)
- **Cross-Validation:** 5-fold stratified cross-validation with metrics:
  - Accuracy, F1 Score, Precision, Recall, MCC, AUC
- **Visualization:**
  - Bar plots for model performance (Accuracy, F1, AUC)
  - Standard deviation plots for model stability
  - Confusion matrices for top models
  - Feature importance for tree-based models
- **Export:** Saves results and summary reports as CSV and TXT files.

## Files

- `FinalProject-Morales-Oro-Partosa.ipynb` — Main notebook with all code, analysis, and visualizations.
- `student_performance_large_dataset.csv` — Input dataset.
- `model_comparison_results_*.csv` — Exported model performance results.
- `model_analysis_summary_*.txt` — Exported summary report.

## Requirements

- Python 3.7+
- Jupyter Notebook or VS Code with Jupyter extension
- Packages: `xgboost`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `numpy`

Install dependencies:

    pip install xgboost scikit-learn matplotlib seaborn pandas numpy

---

## Usage
1. Open the notebook in Jupyter or VS Code.
2. Run all cells in order.
3. Review the printed outputs and visualizations for model comparison and insights.
4. Check the exported CSV and TXT files for detailed results.

---

## Notes
- XGBoost will use GPU acceleration if available; otherwise, it defaults to CPU.
- The notebook is modular: you can toggle inclusion of the Exam_Score feature.
- All results are reproducible with the provided random seeds.

---

## Authors
- Morales, A.
- Oro, D.M.
- Partosa, J. II.

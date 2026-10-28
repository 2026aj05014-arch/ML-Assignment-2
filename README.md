# Loan Approval Classification - ML Assignment 2

## a. Problem Statement

Predict whether a loan application will be approved (`loan_status = 1`) or rejected (`loan_status = 0`) from applicant, financial, and loan attributes. Five classification models are trained on the same dataset, evaluated with six metrics (Accuracy, AUC, Precision, Recall, F1, MCC), saved to disk, and demonstrated through an interactive Streamlit app that evaluates the saved models on a test CSV without retraining.

## b. Dataset Description

| Item | Details |
|---|---|
| Source | Public loan approval dataset (Kaggle) |
| Instances | 45,000 |
| Features | 13 input features + 1 target (`loan_status`) |
| Problem type | Binary classification |
| Class balance | 35,000 (class 0) / 10,000 (class 1) |
| Train / Test split | 80% / 20% stratified, `random_state=42` (36,000 / 9,000 rows) |

**Features**

| Feature | Type | Description |
|---|---|---|
| `person_age` | Numeric | Age of the applicant |
| `person_gender` | Categorical | Gender of the applicant |
| `person_education` | Categorical | Highest education level |
| `person_income` | Numeric | Annual income |
| `person_emp_exp` | Numeric | Years of employment experience |
| `person_home_ownership` | Categorical | RENT / OWN / MORTGAGE / OTHER |
| `loan_amnt` | Numeric | Requested loan amount |
| `loan_intent` | Categorical | Purpose of the loan |
| `loan_int_rate` | Numeric | Loan interest rate |
| `loan_percent_income` | Numeric | Loan amount as a fraction of income |
| `cb_person_cred_hist_length` | Numeric | Credit history length (years) |
| `credit_score` | Numeric | Applicant credit score |
| `previous_loan_defaults_on_file` | Categorical | Whether a previous default exists (Yes / No) |
| `loan_status` | Target | 1 = approved, 0 = rejected |

**Preprocessing** (saved inside every model pipeline, so raw CSV can be fed directly):
- Numeric: median imputation, then `StandardScaler`
- Categorical: most-frequent imputation, then `OneHotEncoder(handle_unknown="ignore")`

## c. GitHub Repository Link

https://github.com/2026aj05014-arch/ML-Assignment-2

**Live Streamlit App:** https://ml-assignment-2-kavin-2026aj05014.streamlit.app/

**Repository structure**

```
.
├── app.py                      # Streamlit application
├── loan_classification.ipynb   # Preprocessing, split, training, evaluation
├── requirements.txt
├── README.md
├── test_data.csv               # Test data used for evaluation / app upload
├── data/                       # Raw and processed train/test splits
├── model/                      # Saved models (.joblib) and preprocessor
└── results/model_metrics.csv   # Evaluation metrics of all models
```

## d. Models Used

All five models were trained on the same training split and evaluated on the same 9,000-row test set.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8993 | 0.9562 | 0.7888 | 0.7470 | 0.7673 | 0.7036 |
| Decision Tree | 0.8977 | 0.8537 | 0.7672 | 0.7745 | 0.7708 | 0.7050 |
| KNN | 0.8956 | 0.9246 | 0.7909 | 0.7205 | 0.7541 | 0.6891 |
| Naive Bayes | 0.7356 | 0.9398 | 0.4566 | 0.9990 | 0.6267 | 0.5484 |
| Support Vector Machine | 0.9178 | 0.9642 | 0.8575 | 0.7555 | 0.8033 | 0.7540 |

Model settings: Logistic Regression (`max_iter=1500`), Decision Tree (default), KNN (`n_neighbors=5`), Gaussian Naive Bayes (default), SVM (RBF kernel). AUC for SVM is computed from `decision_function` scores.

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong and stable baseline: 0.8993 accuracy and 0.9562 AUC, second best AUC overall. Precision (0.7888) is higher than recall (0.7470), so it misses some approvals but makes fewer false approvals. Being linear, it cannot capture non-linear feature interactions, which limits F1 to 0.7673. |
| Decision Tree | Best recall among balanced models (0.7745) and slightly higher F1 than Logistic Regression, but the lowest AUC (0.8537) because tree leaf probabilities are coarse. Default (unpruned) depth likely overfits the training data, so ranking quality is weaker even though accuracy is similar. |
| KNN | Comparable accuracy (0.8956) with good precision (0.7909) but the lowest recall of the balanced models (0.7205). Performance depends on distance in the one-hot and scaled feature space, and inference is slower on large data. F1 (0.7541) is below the top models. |
| Naive Bayes | Weakest overall: accuracy only 0.7356 and precision 0.4566, though recall is almost perfect (0.9990). It labels most applicants as positive, causing many false positives. The independence assumption is violated by correlated features such as income, loan amount, and loan percent of income, and one-hot features do not suit Gaussian likelihoods. Its AUC (0.9398) shows the ranking is good but the default threshold is poorly calibrated. |
| Support Vector Machine | Best model on Accuracy (0.9178), AUC (0.9642), Precision (0.8575), F1 (0.8033), and MCC (0.7540). The RBF kernel captures non-linear boundaries well and gives the best trade-off between precision and recall. Training and prediction are slower than the other models. |
| **Overall, Winner for your dataset?** | **Support Vector Machine** - highest Accuracy, AUC, Precision, F1, and MCC. MCC is the most reliable single metric for this imbalanced dataset (78% / 22%), and SVM leads it clearly (0.7540 vs 0.7050 for the next best). Naive Bayes has the highest recall but at an unacceptable cost in precision. |

## Streamlit App

Features implemented:
1. **Dataset upload (CSV)** - upload a test CSV containing all 13 feature columns plus `loan_status`, or run on the bundled test data.
2. **Model selection dropdown** - choose any of the five saved models.
3. **Evaluation metrics** - Accuracy, AUC, Precision, Recall, F1, MCC.
4. **Confusion matrix and classification report** for the selected model.
5. **All-models comparison table** on the same test data, with CSV download.

Saved models are loaded with `joblib`; no retraining is done on uploaded data.

### Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Sample Test CSV Format

```csv
person_age,person_gender,person_education,person_income,person_emp_exp,person_home_ownership,loan_amnt,loan_intent,loan_int_rate,loan_percent_income,cb_person_cred_hist_length,credit_score,previous_loan_defaults_on_file,loan_status
22.0,female,Master,71948.0,0,RENT,35000.0,PERSONAL,16.02,0.49,3.0,561,No,1
21.0,female,High School,12282.0,0,OWN,1000.0,EDUCATION,11.14,0.08,2.0,504,Yes,0
25.0,female,High School,12438.0,3,MORTGAGE,5500.0,MEDICAL,12.87,0.44,3.0,635,No,1
```

## Reproducing the Results

Run all cells of `loan_classification.ipynb`. It performs preprocessing, the stratified split, model training, saves models to `model/`, and writes metrics to `results/model_metrics.csv`.

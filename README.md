# Loan-Approval-ML-Model

```markdown
# 🏦 Loan Eligibility Prediction Portal

An end-to-end Machine Learning project to automate the classification of loan approval eligibility. This project analyzes applicant demographics and financial history, cleanses the dataset, trains multiple classification models, and serves predictions via an interactive **Streamlit** web application.

## 🚀 Live Demo & Hosting
The web application is configured to run directly on local environments or via secure public tunnels (using `localtunnel` for testing pipelines).

---

## 📊 Dataset Overview
The model is trained on loan applicant details including:
- **Demographics:** Gender, Marital Status, Dependents, Education, Self-Employment, Property Area.
- **Financial Indicators:** Applicant Income, Coapplicant Income, Loan Amount, Loan Term.
- **Credit History:** Credit history presence/absence (highly correlated with eligibility).

### Data Preprocessing & Feature Engineering
1. **Handling Missing Values:** Category modes used for categorical variables; means used for numerical variables.
2. **Log Transformations:** Applied log scaling (`loanAmount_log`, `TotalIncome_log`) to normalize highly skewed distributions.
3. **Scaling:** Applied `StandardScaler` to bring all numerical features onto a uniform scale.
4. **One-Hot Encoding:** Categorical variables converted consistently using standard dummy alignment.

---

## ⚙️ Model Comparisons & Performance
We trained and validated multiple classifiers to identify the best-performing algorithm:

| Model | Accuracy |
| :--- | :---: |
| **Gaussian Naive Bayes (GNB)** | **82.9%** (Best Model) 🏆 |
| Random Forest Classifier | 79.7% |
| K-Nearest Neighbors (KNN) | 75.6% |
| Decision Tree Classifier | 73.1% |

---

## 🛠️ Installation & Setup

To run this project locally, clone this repository and install the dependencies:

```bash
git clone https://github.com/your-username/loan-approval-prediction.git
cd loan-approval-prediction
pip install -r requirements.txt
```

### Running the Streamlit App
Execute the following command to boot up the interactive prediction portal locally:
```bash
streamlit run app.py
```

### Files in this Repository
- `loan_approval.ipynb`: Explorable Jupyter notebook containing the full data science workflow (EDA, modeling, scaling).
- `app.py`: Streamlit python backend routing client inputs to predictions.
- `best_model.pkl`: Serialized Gaussian Naive Bayes classification model.
- `scaler.pkl`: Serialized `StandardScaler` weights for input normalization.
- `model_columns.pkl`: Features list alignment structure.
```

# Bank Fraud Detection

A Flask web application that detects fraudulent bank transactions using **15 machine learning models** trained on real transaction data. Built as part of the Fog and Edge Computing module at National College of Ireland.

---

## Features

- **15 ML models** trained once at startup and cached for fast predictions
- **Majority vote** across all models for final fraud/legitimate verdict
- **Risk score (0–100)** based on transaction signals
- **Live comparison table** showing Accuracy, Precision, Recall, and F1 for all models
- **Bar chart** visualising model performance side by side
- **Class-balanced training** using `class_weight='balanced'` to handle imbalanced fraud data

---

## Models Used

| # | Model |
|---|-------|
| 1 | Logistic Regression |
| 2 | Decision Tree |
| 3 | Random Forest |
| 4 | Gradient Boosting |
| 5 | AdaBoost |
| 6 | Extra Trees |
| 7 | Bagging |
| 8 | K-Nearest Neighbors |
| 9 | SVM (RBF Kernel) |
| 10 | Naive Bayes |
| 11 | Linear Discriminant Analysis (LDA) |
| 12 | Quadratic Discriminant Analysis (QDA) |
| 13 | Ridge Classifier |
| 14 | SGD Classifier |
| 15 | Voting Ensemble (RF + GB + LR) |

---

## Project Structure

```
Fog/
├── app.py                      # Flask application & ML logic
├── bank_transactions_data.csv  # Transaction dataset
├── requirements.txt            # Python dependencies
├── templates/
│   └── index.html              # Jinja2 HTML template
└── static/
    └── style.css               # Application styles
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/PrajwalK2702/Fog.git
cd Fog
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
python app.py
```

### 5. Open in your browser
```
http://127.0.0.1:5000
```

---

## Input Features

| Feature | Description |
|---------|-------------|
| Transaction Amount | Value of the transaction in USD |
| Transaction Type | Debit / Credit / Transfer / Payment |
| Customer Age | Age of the account holder |
| Login Attempts | Number of login attempts before the transaction |
| Account Balance | Current balance of the account |

---

## Fraud Detection Logic

Fraud labels are derived from a rule-based scoring system applied to the dataset:

- Transaction amount > 60% of account balance → +2 points
- Login attempts > 3 → +2 points
- Amount in top 3% of all transactions → +1 point
- Balance in bottom 5% of all balances → +1 point
- Transfer type transaction → +1 point

A score ≥ 3 is labelled as **Fraud**.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_DEBUG` | `false` | Set to `true` to enable Flask debug mode |

```bash
FLASK_DEBUG=true python app.py
```

---

## Tech Stack

- **Backend:** Python, Flask
- **ML:** scikit-learn
- **Data:** pandas, numpy
- **Visualisation:** matplotlib
- **Frontend:** HTML5, CSS3, Jinja2

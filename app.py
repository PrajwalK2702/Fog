from flask import Flask, request, render_template, Response
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, io, copy, warnings, threading
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier,
    BaggingClassifier, VotingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

app = Flask(__name__)

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bank_transactions_data.csv')
DEBUG = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'

_graph_lock = threading.Lock()
_graph_buf  = None
_cache = {'scaler': None, 'models': None, 'Xte': None, 'yte': None}

# ---------------------------------------------------------------------
# UI shows 4 types. They are mapped to 2 training-compatible codes
# that match the actual CSV values (Credit / Debit).
#
#   UI type       | UI code | Training code (CSV-compatible)
#   Credit        |    0    |   0  (Credit in CSV)
#   Debit         |    1    |   1  (Debit  in CSV)
#   Transfer      |    2    |   1  (treated as Debit for training)
#   Payment       |    3    |   1  (treated as Debit for training)
#
# Risk-score logic still uses the full 4-type UI code.
# Final verdict uses accuracy-weighted vote (not simple majority).
# ---------------------------------------------------------------------
UI_TYPES = {
    0: 'Credit',
    1: 'Debit',
    2: 'Transfer',
    3: 'Payment',
}

UI_TO_TRAIN = {0: 0, 1: 1, 2: 1, 3: 1}
CSV_TYPE_MAP = {'Credit': 0, 'Debit': 1}

MODELS_DEF = [
    ("Logistic Regression",  LogisticRegression(max_iter=500, class_weight='balanced')),
    ("Decision Tree",        DecisionTreeClassifier(max_depth=6, class_weight='balanced')),
    ("Random Forest",        RandomForestClassifier(n_estimators=80, class_weight='balanced')),
    ("Gradient Boosting",    GradientBoostingClassifier(n_estimators=80)),
    ("AdaBoost",             AdaBoostClassifier(n_estimators=60)),
    ("Extra Trees",          ExtraTreesClassifier(n_estimators=80, class_weight='balanced')),
    ("Bagging",              BaggingClassifier(n_estimators=60)),
    ("K-Nearest Neighbors",  KNeighborsClassifier(n_neighbors=7)),
    ("SVM (RBF)",            SVC(probability=True, class_weight='balanced')),
    ("Naive Bayes",          GaussianNB()),
    ("LDA",                  LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')),
    ("QDA",                  QuadraticDiscriminantAnalysis(reg_param=0.5)),
    ("Ridge Classifier",     RidgeClassifier(class_weight='balanced')),
    ("SGD Classifier",       SGDClassifier(max_iter=500, loss='modified_huber', class_weight='balanced')),
    ("Voting Ensemble",      None),
]


def load_data():
    df = pd.read_csv(DATA_PATH)
    df['TypeEnc'] = df['TransactionType'].map(CSV_TYPE_MAP)
    if df['TypeEnc'].isna().any():
        unknown = sorted(df.loc[df['TypeEnc'].isna(), 'TransactionType'].astype(str).unique().tolist())
        raise ValueError(f"Unknown CSV transaction types: {unknown}")
    df['TypeEnc'] = df['TypeEnc'].astype(int)

    amt   = df['TransactionAmount']
    bal   = df['AccountBalance']
    login = df['LoginAttempts']
    is_debit = (df['TypeEnc'] == 1).astype(int)

    score = (
        ((amt / (bal + 1)) > 0.6).astype(int) * 2 * is_debit +
        (login > 3).astype(int) * 2 +
        (amt > amt.quantile(0.97)).astype(int) +
        (bal < bal.quantile(0.05)).astype(int)
    )
    df['Fraud'] = (score >= 4).astype(int)
    features = ['TransactionAmount', 'TypeEnc', 'CustomerAge', 'LoginAttempts', 'AccountBalance']
    return df[features].values, df['Fraud'].values


def build_cache():
    print("  Training all 15 models (one-time)...")
    X, y = load_data()

    if int(y.sum()) < X.shape[1]:
        print(f"  WARNING: very few fraud samples, lowering threshold...")
        df = pd.read_csv(DATA_PATH)
        df['TypeEnc'] = df['TransactionType'].map(CSV_TYPE_MAP).astype(int)
        amt   = df['TransactionAmount']
        bal   = df['AccountBalance']
        login = df['LoginAttempts']
        is_debit = (df['TypeEnc'] == 1).astype(int)
        score = (
            ((amt / (bal + 1)) > 0.6).astype(int) * 2 * is_debit +
            (login > 3).astype(int) * 2 +
            (amt > amt.quantile(0.97)).astype(int) +
            (bal < bal.quantile(0.05)).astype(int)
        )
        df['Fraud'] = (score >= 3).astype(int)
        X = df[['TransactionAmount', 'TypeEnc', 'CustomerAge', 'LoginAttempts', 'AccountBalance']].values
        y = df['Fraud'].values
        print(f"  Fraud samples after adjustment: {int(y.sum())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    trained = {}
    for name, clf in MODELS_DEF:
        if clf is None:
            clf = VotingClassifier(
                estimators=[
                    ('rf', trained['Random Forest']),
                    ('gb', trained['Gradient Boosting']),
                    ('lr', trained['Logistic Regression']),
                ],
                voting='soft'
            )
        else:
            clf = copy.deepcopy(clf)
        clf.fit(Xtr, y_train)
        trained[name] = clf

    _cache['scaler'] = scaler
    _cache['models'] = trained
    _cache['Xte']    = Xte
    _cache['yte']    = y_test
    print("  Models ready.")


def run_predictions(input_vec):
    scaler  = _cache['scaler']
    trained = _cache['models']
    Xte     = _cache['Xte']
    yte     = _cache['yte']
    inp     = scaler.transform([input_vec])

    results = []
    for name, _ in MODELS_DEF:
        clf    = trained[name]
        y_pred = clf.predict(Xte)
        pred   = int(clf.predict(inp)[0])
        acc    = accuracy_score(yte, y_pred)
        results.append({
            'name':       name,
            'accuracy':   acc,
            'precision':  precision_score(yte, y_pred, zero_division=0),
            'recall':     recall_score(yte, y_pred, zero_division=0),
            'f1':         f1_score(yte, y_pred, zero_division=0),
            'prediction': pred,
        })
    results.sort(key=lambda r: r['accuracy'], reverse=True)
    return results


def weighted_verdict(report):
    """
    Accuracy-weighted vote.
    Each model contributes its accuracy as its vote weight.
    If weighted fraud score > 0.5 * total weight → Fraud.
    Also returns fraud_weight_pct (0-100) used to boost risk score.
    """
    total_weight  = sum(r['accuracy'] for r in report)
    fraud_weight  = sum(r['accuracy'] for r in report if r['prediction'] == 1)
    fraud_pct     = fraud_weight / total_weight if total_weight > 0 else 0
    is_fraud      = fraud_pct > 0.5
    return is_fraud, round(fraud_pct * 100, 1)


def make_graph(report_data):
    global _graph_buf
    names = [r['name']      for r in report_data]
    accs  = [r['accuracy']  for r in report_data]
    precs = [r['precision'] for r in report_data]
    recs  = [r['recall']    for r in report_data]
    f1s   = [r['f1']        for r in report_data]

    x, w = np.arange(len(names)), 0.2
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('#131929')
    ax.set_facecolor('#1a2236')
    ax.bar(x - 1.5*w, accs,  w, label='Accuracy',  color='#38bdf8', alpha=0.85)
    ax.bar(x - 0.5*w, precs, w, label='Precision', color='#4ade80', alpha=0.85)
    ax.bar(x + 0.5*w, recs,  w, label='Recall',    color='#fbbf24', alpha=0.85)
    ax.bar(x + 1.5*w, f1s,   w, label='F1 Score',  color='#f87171', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha='right', color='#94a3b8', fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Score', color='#94a3b8')
    ax.tick_params(colors='#94a3b8')
    ax.spines[:].set_color('#1e293b')
    ax.yaxis.grid(True, color='#1e293b', linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(facecolor='#0b0f1a', edgecolor='#1e293b', labelcolor='#e2e8f0', fontsize=9)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    buf.seek(0)
    with _graph_lock:
        _graph_buf = buf.read()


@app.route('/graph')
def serve_graph():
    with _graph_lock:
        data = _graph_buf
    if data is None:
        return Response(status=204)
    return Response(data, mimetype='image/png')


@app.route('/', methods=['GET', 'POST'])
def index():
    ctx = dict(
        result=None, detail=None, best_model=None, best_acc=None,
        report_data=None, error=None,
        amount='', sel_type='', age='', login='', balance=''
    )

    if request.method == 'POST':
        ctx['amount']   = request.form.get('amount', '')
        ctx['sel_type'] = request.form.get('type', '')
        ctx['age']      = request.form.get('age', '')
        ctx['login']    = request.form.get('login', '')
        ctx['balance']  = request.form.get('balance', '')

        try:
            amount  = float(ctx['amount'])
            ui_type = int(ctx['sel_type'])
            age     = int(ctx['age'])
            login   = int(ctx['login'])
            balance = float(ctx['balance'])

            if amount <= 0:
                raise ValueError("Transaction amount must be greater than zero.")
            if balance < 0:
                raise ValueError("Account balance cannot be negative.")
            if age < 18 or age > 100:
                raise ValueError("Customer age must be between 18 and 100.")
            if login < 0:
                raise ValueError("Login attempts cannot be negative.")
            if ui_type not in UI_TYPES:
                raise ValueError("Invalid transaction type selected.")

            train_type = UI_TO_TRAIN[ui_type]
            input_vec  = [amount, train_type, age, login, balance]
            report     = run_predictions(input_vec)

            best = report[0]
            ctx['best_model'] = best['name']
            ctx['best_acc']   = best['accuracy']

            # ── Accuracy-weighted verdict ──────────────────────────────
            is_fraud, fraud_weight_pct = weighted_verdict(report)
            ctx['result']           = "Fraud Detected" if is_fraud else "Legitimate Transaction"
            ctx['fraud_weight_pct'] = fraud_weight_pct

            ratio = round((amount / (balance + 1)) * 100, 1)

            # ── Rule-based risk signals ────────────────────────────────
            if ui_type == 0:
                ratio_penalty = 0
            elif ui_type == 3:
                ratio_penalty = 0
            elif ui_type == 2:
                ratio_penalty = 25 if ratio > 150 else 0
            else:
                ratio_penalty = 20 if ratio > 80 else 0

            login_risk       = min(max(login - 3, 0) * 5, 35) if login > 3 else 0
            low_balance_risk = 10 if balance < 500 else 0
            high_amount_risk = 20 if amount > 10000 else (10 if amount > 5000 else 0)

            # ── ML consensus boost ────────────────────────────────────
            # If the weighted ML vote is > 50% fraud, add up to +30 pts
            # proportional to how strong the consensus is (50%→0, 100%→30).
            ml_boost = 0
            if fraud_weight_pct > 50:
                ml_boost = int(((fraud_weight_pct - 50) / 50) * 30)

            raw_score = ratio_penalty + login_risk + high_amount_risk + low_balance_risk + ml_boost

            ctx['detail'] = {
                'amount':     f"{amount:,.2f}",
                'balance':    f"{balance:,.2f}",
                'ratio':      ratio,
                'login':      login,
                'type':       UI_TYPES[ui_type],
                'risk_score': min(100, raw_score),
                'ml_boost':   ml_boost,
            }
            ctx['report_data'] = report
            make_graph(report)

        except ValueError as e:
            ctx['error'] = str(e)
        except Exception as e:
            ctx['error'] = f"Unexpected error: {str(e)}"

    return render_template('index.html', **ctx)


if __name__ == '__main__':
    print("\n  Bank Fraud Detection server starting...")
    build_cache()
    print("  Open this in your browser: http://127.0.0.1:5000\n")
    app.run(debug=DEBUG)

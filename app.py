from flask import Flask, request, render_template, Response
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, io, copy, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

# ── Config ───────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bank_transactions_data.csv')
GRAPH_BUF = None

MODELS_DEF = [
    ("Logistic Regression",  LogisticRegression(max_iter=500)),
    ("Decision Tree",        DecisionTreeClassifier(max_depth=6)),
    ("Random Forest",        RandomForestClassifier(n_estimators=80)),
    ("Gradient Boosting",    GradientBoostingClassifier(n_estimators=80)),
    ("AdaBoost",             AdaBoostClassifier(n_estimators=60)),
    ("Extra Trees",          ExtraTreesClassifier(n_estimators=80)),
    ("Bagging",              BaggingClassifier(n_estimators=60)),
    ("K-Nearest Neighbors",  KNeighborsClassifier(n_neighbors=7)),
    ("SVM (RBF)",            SVC(probability=True)),
    ("Naive Bayes",          GaussianNB()),
    ("LDA",                  LinearDiscriminantAnalysis()),
    ("QDA",                  QuadraticDiscriminantAnalysis()),
    ("Ridge Classifier",     RidgeClassifier()),
    ("SGD Classifier",       SGDClassifier(max_iter=500, loss='modified_huber')),
    ("Voting Ensemble",      None),
]


# ── Data helpers ─────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH)
    le = LabelEncoder()
    df['TypeEnc'] = le.fit_transform(df['TransactionType'].astype(str))
    amt   = df['TransactionAmount']
    bal   = df['AccountBalance']
    login = df['LoginAttempts']
    score = (
        ((amt / (bal + 1)) > 0.6).astype(int) * 2 +
        (login > 3).astype(int) * 2 +
        (amt > amt.quantile(0.97)).astype(int) +
        (bal < bal.quantile(0.05)).astype(int) +
        (df['TypeEnc'] == 2).astype(int)
    )
    df['Fraud'] = (score >= 3).astype(int)
    features = ['TransactionAmount', 'TypeEnc', 'CustomerAge', 'LoginAttempts', 'AccountBalance']
    return df[features].values, df['Fraud'].values


# ── Model runner ─────────────────────────────────────────────────────────────
def run_models(X, y, input_vec):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=np.random.randint(0, 9999)
    )
    scaler = StandardScaler()
    Xtr    = scaler.fit_transform(X_train)
    Xte    = scaler.transform(X_test)
    inp    = scaler.transform([input_vec])

    results, trained = [], {}

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
        clf = copy.deepcopy(clf)
        clf.fit(Xtr, y_train)
        y_pred = clf.predict(Xte)
        pred   = int(clf.predict(inp)[0])
        trained[name] = clf

        results.append({
            'name':       name,
            'accuracy':   accuracy_score(y_test, y_pred),
            'precision':  precision_score(y_test, y_pred, zero_division=0),
            'recall':     recall_score(y_test, y_pred, zero_division=0),
            'f1':         f1_score(y_test, y_pred, zero_division=0),
            'prediction': pred,
        })

    results.sort(key=lambda r: r['accuracy'], reverse=True)
    return results


# ── Graph generator ──────────────────────────────────────────────────────────
def make_graph(report_data):
    global GRAPH_BUF
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
    GRAPH_BUF = buf.read()


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/graph')
def serve_graph():
    if GRAPH_BUF is None:
        return Response(status=204)
    return Response(GRAPH_BUF, mimetype='image/png')


@app.route('/', methods=['GET', 'POST'])
def index():
    global GRAPH_BUF
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
            t_type  = int(ctx['sel_type'])
            age     = int(ctx['age'])
            login   = int(ctx['login'])
            balance = float(ctx['balance'])

            X, y      = load_data()
            input_vec = [amount, t_type, age, login, balance]
            report    = run_models(X, y, input_vec)

            best              = report[0]
            ctx['best_model'] = best['name']
            ctx['best_acc']   = best['accuracy']

            votes    = sum(1 for r in report if r['prediction'] == 1)
            is_fraud = votes > len(report) / 2
            ctx['result'] = "Fraud Detected" if is_fraud else "Legitimate Transaction"

            ratio = round((amount / (balance + 0.01)) * 100, 1)
            ctx['detail'] = {
                'amount':     f"{amount:,.2f}",
                'balance':    f"{balance:,.2f}",
                'ratio':      ratio,
                'login':      login,
                'risk_score': min(100, int(
                    (ratio > 60) * 30 +
                    (login > 3)  * 30 +
                    (amount > 5000) * 20 +
                    (balance < 500) * 20
                )),
            }
            ctx['report_data'] = report
            make_graph(report)

        except Exception as e:
            ctx['error'] = str(e)

    return render_template('index.html', **ctx)


if __name__ == '__main__':
    print("\n  Bank Fraud Detection server starting...")
    print("  Open this in your browser: http://127.0.0.1:5000\n")
    app.run(debug=True)

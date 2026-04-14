from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier, ExtraTreesClassifier,
                               BaggingClassifier, VotingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

app = Flask(__name__)

# ── HTML template (embedded — no templates/ folder needed) ──────────────────
HTML = """<!DOCTYPE html>
<html>
<head>
<title>Bank Fraud Detection</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #0b0f1a; --surface: #131929; --surface2: #1a2236;
    --border: rgba(255,255,255,0.07); --border-accent: rgba(56,189,248,0.3);
    --blue: #38bdf8; --blue-dim: rgba(56,189,248,0.12);
    --green: #4ade80; --green-dim: rgba(74,222,128,0.08);
    --red: #f87171;   --red-dim: rgba(248,113,113,0.08);
    --amber: #fbbf24;
    --text: #e2e8f0; --text-muted: #64748b; --text-dim: #94a3b8;
    --mono: 'DM Mono', monospace; --sans: 'DM Sans', sans-serif;
  }
  body { font-family: var(--sans); background: var(--bg); color: var(--text); min-height: 100vh; padding: 36px 20px; }
  .page { max-width: 1100px; margin: 0 auto; }

  .header { display: flex; align-items: center; gap: 14px; margin-bottom: 32px; }
  .hicon { width: 40px; height: 40px; background: var(--blue-dim); border: 1px solid var(--border-accent); border-radius: 10px; display: flex; align-items: center; justify-content: center; }
  .hicon svg { width: 20px; height: 20px; stroke: var(--blue); fill: none; stroke-width: 1.8; }
  .header h1 { font-size: 20px; font-weight: 500; letter-spacing: -0.3px; }
  .header p { font-size: 13px; color: var(--text-muted); margin-top: 2px; }

  .top { display: grid; grid-template-columns: 300px 1fr; gap: 14px; align-items: start; }
  @media(max-width:680px){ .top { grid-template-columns: 1fr; } }

  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 22px; }
  .card-title { font-family: var(--mono); font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 16px; }

  .field { margin-bottom: 12px; }
  .field label { display: block; font-size: 12px; color: var(--text-dim); margin-bottom: 5px; }
  .field input, .field select {
    width: 100%; background: var(--surface2); border: 1px solid var(--border);
    border-radius: 8px; color: var(--text); font-family: var(--mono); font-size: 14px;
    padding: 9px 11px; outline: none; transition: border-color 0.2s; -webkit-appearance: none;
  }
  .field input::placeholder { color: var(--text-muted); }
  .field input:focus, .field select:focus { border-color: var(--border-accent); }
  .field select option { background: var(--surface2); }
  .submit-btn {
    width: 100%; background: var(--blue); color: #0b0f1a; border: none;
    border-radius: 8px; font-family: var(--sans); font-size: 14px; font-weight: 600;
    padding: 12px; cursor: pointer; margin-top: 6px; transition: opacity 0.2s, transform 0.1s;
  }
  .submit-btn:hover { opacity: 0.85; }
  .submit-btn:active { transform: scale(0.98); }

  .result-box { border-radius: 12px; padding: 18px 20px; border: 1px solid transparent; }
  .result-box.safe  { background: var(--green-dim); border-color: rgba(74,222,128,0.25); }
  .result-box.fraud { background: var(--red-dim);   border-color: rgba(248,113,113,0.25); }
  .verdict { font-size: 20px; font-weight: 600; display: flex; align-items: center; gap: 9px; margin-bottom: 4px; }
  .safe  .verdict { color: var(--green); }
  .fraud .verdict { color: var(--red); }
  .verdict-sub { font-size: 13px; color: var(--text-dim); margin-bottom: 14px; }
  .dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
  .dot-g { background: var(--green); }
  .dot-r { background: var(--red); }

  .risk-track { height: 6px; background: rgba(255,255,255,0.07); border-radius: 4px; overflow: hidden; margin-bottom: 5px; }
  .risk-fill  { height: 100%; border-radius: 4px; transition: width 0.5s ease; }
  .risk-label { font-family: var(--mono); font-size: 11px; color: var(--text-muted); margin-bottom: 14px; }

  .detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 14px; }
  .di { background: rgba(255,255,255,0.04); border-radius: 7px; padding: 8px 10px; }
  .di .dl { font-size: 11px; color: var(--text-muted); margin-bottom: 2px; }
  .di .dv { font-family: var(--mono); font-size: 14px; font-weight: 500; }

  .best-badge {
    display: inline-flex; align-items: center; gap: 6px;
    font-family: var(--mono); font-size: 12px;
    background: var(--blue-dim); color: var(--blue);
    border: 1px solid var(--border-accent); border-radius: 20px; padding: 4px 12px;
  }

  .empty { padding: 40px 20px; text-align: center; color: var(--text-muted); font-size: 13px; line-height: 2; }
  .empty-arrow { font-size: 22px; margin-bottom: 8px; }

  .section-label { font-family: var(--mono); font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.12em; margin: 26px 0 10px; }

  .tbl-wrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  thead th { font-family: var(--mono); font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; text-align: left; padding: 8px 13px; border-bottom: 1px solid var(--border); font-weight: 400; }
  tbody tr { border-bottom: 1px solid var(--border); transition: background 0.12s; }
  tbody tr:hover { background: var(--surface2); }
  tbody tr:last-child { border-bottom: none; }
  tbody tr.best { background: rgba(56,189,248,0.05); }
  td { padding: 10px 13px; font-family: var(--mono); font-size: 13px; }
  td.nm { font-family: var(--sans); font-size: 13px; color: var(--text-dim); }
  td.nm .star { color: var(--blue); margin-left: 5px; font-size: 12px; }

  .pill { display: inline-block; font-family: var(--mono); font-size: 12px; padding: 2px 8px; border-radius: 20px; }
  .ph  { background: rgba(74,222,128,0.12);  color: #4ade80; }
  .pm  { background: rgba(251,191,36,0.12);  color: #fbbf24; }
  .pl  { background: rgba(248,113,113,0.12); color: #f87171; }

  .graph-card { margin-top: 14px; }
  .graph-card img { width: 100%; border-radius: 10px; border: 1px solid var(--border); display: block; margin-top: 4px; }

  .error-box { background: rgba(248,113,113,0.08); border: 1px solid rgba(248,113,113,0.25); border-radius: 12px; padding: 18px 20px; color: #f87171; font-family: var(--mono); font-size: 13px; }
</style>
</head>
<body>
<div class="page">

  <div class="header">
    <div class="hicon">
      <svg viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.955 11.955 0 003 12c0 6.627 5.373 12 12 12s12-5.373 12-12c0-2.03-.504-3.942-1.394-5.617"/></svg>
    </div>
    <div>
      <h1>Bank Fraud Detection</h1>
      <p>Every submission re-trains all 15 models — results change with each input</p>
    </div>
  </div>

  <div class="top">
    <!-- FORM -->
    <div class="card">
      <div class="card-title">Transaction input</div>
      <form method="POST">
        <div class="field">
          <label>Transaction amount ($)</label>
          <input name="amount" type="number" step="0.01" placeholder="e.g. 450.00" required value="{{ amount }}">
        </div>
        <div class="field">
          <label>Transaction type</label>
          <select name="type" required>
            <option value="" disabled {% if not sel_type %}selected{% endif %}>Select type</option>
            <option value="0" {% if sel_type=='0' %}selected{% endif %}>0 — Debit</option>
            <option value="1" {% if sel_type=='1' %}selected{% endif %}>1 — Credit</option>
            <option value="2" {% if sel_type=='2' %}selected{% endif %}>2 — Transfer</option>
            <option value="3" {% if sel_type=='3' %}selected{% endif %}>3 — Payment</option>
          </select>
        </div>
        <div class="field">
          <label>Customer age</label>
          <input name="age" type="number" placeholder="e.g. 34" min="18" max="100" required value="{{ age }}">
        </div>
        <div class="field">
          <label>Login attempts</label>
          <input name="login" type="number" placeholder="e.g. 1" min="1" required value="{{ login }}">
        </div>
        <div class="field">
          <label>Account balance ($)</label>
          <input name="balance" type="number" step="0.01" placeholder="e.g. 3200.00" required value="{{ balance }}">
        </div>
        <button type="submit" class="submit-btn">Run prediction</button>
      </form>
    </div>

    <!-- RESULT -->
    <div>
      {% if error %}
        <div class="error-box">⚠ {{ error }}</div>
      {% elif result is not none %}
        {% set is_fraud = "Fraud" in result %}
        <div class="result-box {{ 'fraud' if is_fraud else 'safe' }}">
          <div class="verdict">
            <span class="dot {{ 'dot-r' if is_fraud else 'dot-g' }}"></span>
            {{ result }}
          </div>
          <div class="verdict-sub">
            {% if is_fraud %}High-risk signals detected — block and escalate for review.
            {% else %}Transaction appears legitimate. Low risk signals detected.{% endif %}
          </div>

          <div class="risk-track">
            <div class="risk-fill" style="width:{{ detail.risk_score }}%;
              background:{% if detail.risk_score >= 60 %}#f87171{% elif detail.risk_score >= 35 %}#fbbf24{% else %}#4ade80{% endif %};"></div>
          </div>
          <div class="risk-label">Risk score: {{ detail.risk_score }}/100</div>

          <div class="detail-grid">
            <div class="di"><div class="dl">Amount</div><div class="dv">${{ detail.amount }}</div></div>
            <div class="di"><div class="dl">Balance</div><div class="dv">${{ detail.balance }}</div></div>
            <div class="di"><div class="dl">Amt / Balance</div><div class="dv">{{ detail.ratio }}%</div></div>
            <div class="di"><div class="dl">Login attempts</div><div class="dv">{{ detail.login }}</div></div>
          </div>

          <div class="best-badge">&#9733; Best model: {{ best_model }} &nbsp;|&nbsp; acc {{ (best_acc * 100)|round(1) }}%</div>
        </div>
      {% else %}
        <div class="card">
          <div class="empty">
            <div class="empty-arrow">&#8592;</div>
            Fill in the transaction details<br>and click <strong>Run prediction</strong><br>
            to see live results and all 15 model scores.
          </div>
        </div>
      {% endif %}
    </div>
  </div>

  {% if report_data %}
  <div class="section-label">All 15 models — sorted by accuracy</div>
  <div class="card">
    <div class="tbl-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th><th>Model</th><th>Accuracy</th>
            <th>Precision</th><th>Recall</th><th>F1 Score</th>
          </tr>
        </thead>
        <tbody>
          {% for row in report_data %}
          <tr {% if row.name == best_model %}class="best"{% endif %}>
            <td>{{ loop.index }}</td>
            <td class="nm">{{ row.name }}{% if row.name == best_model %}<span class="star">&#9733;</span>{% endif %}</td>
            <td><span class="pill {% if row.accuracy >= 0.9 %}ph{% elif row.accuracy >= 0.7 %}pm{% else %}pl{% endif %}">{{ (row.accuracy*100)|round(1) }}%</span></td>
            <td><span class="pill {% if row.precision >= 0.9 %}ph{% elif row.precision >= 0.7 %}pm{% else %}pl{% endif %}">{{ (row.precision*100)|round(1) }}%</span></td>
            <td><span class="pill {% if row.recall >= 0.9 %}ph{% elif row.recall >= 0.7 %}pm{% else %}pl{% endif %}">{{ (row.recall*100)|round(1) }}%</span></td>
            <td><span class="pill {% if row.f1 >= 0.9 %}ph{% elif row.f1 >= 0.7 %}pm{% else %}pl{% endif %}">{{ (row.f1*100)|round(1) }}%</span></td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  </div>

  <div class="card graph-card">
    <div class="card-title">Model comparison graph</div>
    <img src="/graph" alt="Model Accuracy Graph">
  </div>
  {% endif %}

</div>
</body>
</html>"""


# ── Data & Model helpers ─────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bank_transactions_data.csv')

GRAPH_BUF = None   # store graph bytes in memory (no static/ folder needed)

MODELS_DEF = [
    ("Logistic Regression",   LogisticRegression(max_iter=500)),
    ("Decision Tree",         DecisionTreeClassifier(max_depth=6)),
    ("Random Forest",         RandomForestClassifier(n_estimators=80)),
    ("Gradient Boosting",     GradientBoostingClassifier(n_estimators=80)),
    ("AdaBoost",              AdaBoostClassifier(n_estimators=60)),
    ("Extra Trees",           ExtraTreesClassifier(n_estimators=80)),
    ("Bagging",               BaggingClassifier(n_estimators=60)),
    ("K-Nearest Neighbors",   KNeighborsClassifier(n_neighbors=7)),
    ("SVM (RBF)",             SVC(probability=True)),
    ("Naive Bayes",           GaussianNB()),
    ("LDA",                   LinearDiscriminantAnalysis()),
    ("QDA",                   QuadraticDiscriminantAnalysis()),
    ("Ridge Classifier",      RidgeClassifier()),
    ("SGD Classifier",        SGDClassifier(max_iter=500, loss='modified_huber')),
    ("Voting Ensemble",       None),
]


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


def run_models(X, y, input_vec):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=np.random.randint(0, 9999)
    )
    scaler    = StandardScaler()
    Xtr       = scaler.fit_transform(X_train)
    Xte       = scaler.transform(X_test)
    inp       = scaler.transform([input_vec])

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
        import copy
        clf = copy.deepcopy(clf)
        clf.fit(Xtr, y_train)
        y_pred = clf.predict(Xte)
        pred   = int(clf.predict(inp)[0])
        trained[name] = clf

        results.append({
            'name':      name,
            'accuracy':  accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall':    recall_score(y_test, y_pred, zero_division=0),
            'f1':        f1_score(y_test, y_pred, zero_division=0),
            'prediction': pred,
        })

    results.sort(key=lambda r: r['accuracy'], reverse=True)
    return results


def make_graph(report_data):
    global GRAPH_BUF
    import io
    names = [r['name'] for r in report_data]
    accs  = [r['accuracy']  for r in report_data]
    precs = [r['precision'] for r in report_data]
    recs  = [r['recall']    for r in report_data]
    f1s   = [r['f1']        for r in report_data]

    x = np.arange(len(names))
    w = 0.2

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
    from flask import Response
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

            best           = report[0]
            ctx['best_model'] = best['name']
            ctx['best_acc']   = best['accuracy']

            votes    = sum(1 for r in report if r['prediction'] == 1)
            is_fraud = votes > len(report) / 2
            ctx['result'] = "🚨 Fraud Detected" if is_fraud else "✅ Legitimate Transaction"

            ratio = round((amount / (balance + 0.01)) * 100, 1)
            ctx['detail'] = {
                'amount':     f"{amount:,.2f}",
                'balance':    f"{balance:,.2f}",
                'ratio':      ratio,
                'login':      login,
                'risk_score': min(100, int(
                    (ratio > 60) * 30 +
                    (login > 3) * 30 +
                    (amount > 5000) * 20 +
                    (balance < 500) * 20
                )),
            }
            ctx['report_data'] = report
            make_graph(report)

        except Exception as e:
            ctx['error'] = str(e)

    return render_template_string(HTML, **ctx)


if __name__ == '__main__':
    print("\n✅  Bank Fraud Detection server starting...")
    print("👉  Open this in your browser: http://127.0.0.1:5000\n")
    app.run(debug=True)

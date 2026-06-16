# app.py  — ChurnGuard AI  |  Full-featured Flask application
from flask import (Flask, render_template, request, jsonify,
                   session, redirect, url_for, send_file)
import pandas as pd
import numpy as np
import joblib
import os, io, json
from datetime import datetime

from feature_utils import add_features
from database import init_db, save_prediction, get_recent_predictions, get_prediction_stats, db
# from shap_explainer import compute_shap_values, format_feature_name
try:
    from shap_explainer import compute_shap_values, format_feature_name
    SHAP_AVAILABLE = True

except Exception as e:

    print(f"⚠️ SHAP disabled: {e}")

    SHAP_AVAILABLE = False

    def compute_shap_values(*args, **kwargs):
        return {
            'top_positive': [],
            'top_negative': [],
            'success': False
        }

    def format_feature_name(name):
        return name
from ai_recommendations import get_ai_recommendation
from segmentation import run_segmentation
from pdf_report import generate_pdf

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'churnguard-dev-secret-change-in-prod')
init_db(app)

MODEL_PATH       = 'best_churn_model.pkl'
COLUMN_INFO_PATH = 'column_info.pkl'
model            = None
column_info      = None

def load_model():
    global model, column_info
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print("✅ Model loaded")
        if os.path.exists(COLUMN_INFO_PATH):
            column_info = joblib.load(COLUMN_INFO_PATH)
        return True
    except Exception as e:
        print(f"❌ {e}")
        return False

load_model()

GENDER_OPTIONS     = ['Male', 'Female']
YES_NO_OPTIONS     = ['Yes', 'No']
YES_NO_NO_SERVICE  = ['Yes', 'No', 'No phone service']
YES_NO_NO_INTERNET = ['Yes', 'No', 'No internet service']
INTERNET_OPTIONS   = ['DSL', 'Fiber optic', 'No']
CONTRACT_OPTIONS   = ['Month-to-month', 'One year', 'Two year']
PAYMENT_OPTIONS    = ['Electronic check', 'Mailed check',
                      'Bank transfer (automatic)', 'Credit card (automatic)']

# ── Core routes ───────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            customer_data = {
                'gender':           request.form['gender'],
                'SeniorCitizen':    int(request.form['SeniorCitizen']),
                'Partner':          request.form['Partner'],
                'Dependents':       request.form['Dependents'],
                'tenure':           int(request.form['tenure']),
                'PhoneService':     request.form['PhoneService'],
                'MultipleLines':    request.form['MultipleLines'],
                'InternetService':  request.form['InternetService'],
                'OnlineSecurity':   request.form['OnlineSecurity'],
                'OnlineBackup':     request.form['OnlineBackup'],
                'DeviceProtection': request.form['DeviceProtection'],
                'TechSupport':      request.form['TechSupport'],
                'StreamingTV':      request.form.get('StreamingTV', 'No'),
                'StreamingMovies':  request.form.get('StreamingMovies', 'No'),
                'Contract':         request.form['Contract'],
                'PaperlessBilling': request.form['PaperlessBilling'],
                'PaymentMethod':    request.form['PaymentMethod'],
                'MonthlyCharges':   float(request.form['MonthlyCharges']),
                'TotalCharges':     float(request.form['TotalCharges']),
            }
            input_df = add_features(pd.DataFrame([customer_data]))
            if model is None:
                return _predict_form("Model not loaded. Run model.py first.")

            probability = float(model.predict_proba(input_df)[0, 1])
            prediction  = int(model.predict(input_df)[0])

            shap_data  = compute_shap_values(model, input_df)
            ai_result  = get_ai_recommendation(
                customer_data, probability, shap_data,
                api_key=os.environ.get('ANTHROPIC_API_KEY')
            )

            with app.app_context():
                record = save_prediction(customer_data, probability, prediction,
                                         shap_values=shap_data,
                                         ai_recommendation=ai_result.get('recommendation', ''))
                record_id = record.id

            session['prediction_result'] = {
                'probability':       probability,
                'prediction':        prediction,
                'customer_data':     customer_data,
                'shap_data':         shap_data,
                'ai_recommendation': ai_result.get('recommendation', ''),
                'ai_success':        ai_result.get('success', False),
                'record_id':         record_id,
            }
            return redirect(url_for('result'))

        except Exception as e:
            return _predict_form(f"Error: {str(e)}")

    return _predict_form()


def _predict_form(error=None):
    return render_template('predict.html', error=error,
                           gender_options=GENDER_OPTIONS,
                           yes_no_options=YES_NO_OPTIONS,
                           internet_options=INTERNET_OPTIONS,
                           contract_options=CONTRACT_OPTIONS,
                           payment_options=PAYMENT_OPTIONS,
                           yes_no_no_service=YES_NO_NO_SERVICE,
                           yes_no_no_internet=YES_NO_NO_INTERNET)


@app.route('/result')
def result():
    result_data = session.get('prediction_result')
    if not result_data:
        return redirect(url_for('predict'))
    recommendations = _generate_recommendations(result_data['probability'])
    return render_template('result.html', result=result_data,
                           recommendations=recommendations)


@app.route('/dashboard')
def dashboard():
    with app.app_context():
        stats  = get_prediction_stats()
        recent = [p.to_dict() for p in get_recent_predictions(10)]
    return render_template('dashboard.html', stats=stats, logs=recent)


# ── SHAP ──────────────────────────────────────────────────────────────

@app.route('/api/shap', methods=['POST'])
def api_shap():
    try:
        data = request.get_json()
        df   = add_features(pd.DataFrame([data]))
        res  = compute_shap_values(model, df)
        res['top_positive'] = [(format_feature_name(f), v)
                               for f, v in res.get('top_positive', [])]
        res['top_negative'] = [(format_feature_name(f), v)
                               for f, v in res.get('top_negative', [])]
        return jsonify(res)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── Claude AI Recommendation ──────────────────────────────────────────

@app.route('/api/ai-recommend', methods=['POST'])
def api_ai_recommend():
    try:
        body        = request.get_json()
        api_key     = body.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
        result      = get_ai_recommendation(
            body.get('customer_data', {}),
            float(body.get('probability', 0.5)),
            body.get('shap_data'),
            api_key
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── Customer Segmentation ─────────────────────────────────────────────

@app.route('/segmentation')
def segmentation_page():
    try:
        n = max(2, min(int(request.args.get('n_clusters', 4)), 7))
        if not os.path.exists('Telco_Churn_Cleaned.csv'):
            return render_template('segmentation.html',
                                   error="Dataset not found.", segments=[],
                                   pca_points='[]', n_clusters=4)
        res = run_segmentation('Telco_Churn_Cleaned.csv', n_clusters=n)
        return render_template('segmentation.html',
                               segments=res['segments'],
                               pca_points=json.dumps(res['pca_points']),
                               n_clusters=n, features=res['features'], error=None)
    except Exception as e:
        return render_template('segmentation.html',
                               error=str(e), segments=[],
                               pca_points='[]', n_clusters=4)


@app.route('/api/segmentation')
def api_segmentation():
    try:
        n   = max(2, min(int(request.args.get('n_clusters', 4)), 7))
        res = run_segmentation('Telco_Churn_Cleaned.csv', n_clusters=n)
        return jsonify({'success': True, **res})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── PDF Reports ───────────────────────────────────────────────────────

@app.route('/report/download', methods=['POST'])
def download_report():
    try:
        r = session.get('prediction_result')
        if not r:
            return jsonify({'error': 'No prediction in session'}), 400
        pdf = generate_pdf(r['customer_data'], r['probability'], r['prediction'],
                           r.get('shap_data'), r.get('ai_recommendation'),
                           r.get('record_id'))
        return send_file(io.BytesIO(pdf), mimetype='application/pdf',
                         as_attachment=True,
                         download_name=f"churn_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/report/download/<int:record_id>')
def download_report_by_id(record_id):
    try:
        from database import Prediction
        rec = Prediction.query.get_or_404(record_id)
        pdf = generate_pdf(
            json.loads(rec.customer_data), rec.probability, rec.prediction,
            json.loads(rec.shap_values) if rec.shap_values else None,
            rec.ai_recommendation, rec.id
        )
        return send_file(io.BytesIO(pdf), mimetype='application/pdf',
                         as_attachment=True,
                         download_name=f"churn_report_{record_id}.pdf")
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Database API ──────────────────────────────────────────────────────

@app.route('/api/predictions')
def api_predictions():
    limit = int(request.args.get('limit', 50))
    preds = [p.to_dict() for p in get_recent_predictions(limit)]
    return jsonify({'success': True, 'predictions': preds, 'total': len(preds)})


@app.route('/api/predictions/<int:record_id>')
def api_prediction_detail(record_id):
    from database import Prediction
    rec  = Prediction.query.get_or_404(record_id)
    data = rec.to_dict()
    data['shap_data']         = json.loads(rec.shap_values) if rec.shap_values else None
    data['ai_recommendation'] = rec.ai_recommendation
    return jsonify(data)


# ── Original API predict ──────────────────────────────────────────────

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        df   = add_features(pd.DataFrame([data]))
        if model is None:
            return jsonify({'success': False, 'error': 'Model not loaded'})
        prob = float(model.predict_proba(df)[0, 1])
        pred = int(model.predict(df)[0])
        return jsonify({'success': True, 'probability': prob,
                        'prediction': 'Churn' if pred else 'No Churn',
                        'risk_level': 'High' if prob > 0.7 else 'Medium' if prob > 0.4 else 'Low'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── Genetic Algorithm Routes (preserved) ─────────────────────────────

@app.route('/ga-optimize')
def ga_optimize_page():
    history, best_params, evolution_log = [], {}, []
    if os.path.exists('ga_history.json'):
        with open('ga_history.json') as f:
            history = json.load(f)
        if history:
            best_params = history[-1].get('best_params', {})
    if os.path.exists('ga_evolution_log.csv'):
        df = pd.read_csv('ga_evolution_log.csv')
        evolution_log = df[['generation', 'best_fitness', 'mean_fitness']].to_dict('records')
    return render_template('ga_results.html', history=history,
                           best_params=best_params,
                           evolution_log=json.dumps(evolution_log),
                           has_results=bool(history))


@app.route('/api/ga-history')
def api_ga_history():
    if os.path.exists('ga_history.json'):
        with open('ga_history.json') as f:
            return jsonify(json.load(f))
    return jsonify([])


@app.route('/api/ga-run', methods=['POST'])
def api_ga_run():
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from genetic_algorithm import run_ga_optimization

        df = pd.read_csv('Telco_Churn_Cleaned.csv')
        le = LabelEncoder()
        df['Churn'] = le.fit_transform(df['Churn'].astype(str))
        X = add_features(df.drop('Churn', axis=1))
        y = df['Churn']
        if 'Tenure_Group' in X.columns:
            X['Tenure_Group'] = X['Tenure_Group'].astype(str)

        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        preprocessor = ColumnTransformer(transformers=[
            ('num', Pipeline([('scaler', StandardScaler())]), num_cols),
            ('cat', Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore',
                                                       sparse_output=False))]), cat_cols),
        ])
        body     = request.get_json() or {}
        ga, best = run_ga_optimization(X_train, y_train, preprocessor,
                                       int(body.get('population_size', 10)),
                                       int(body.get('n_generations', 8)),
                                       save_history=True)
        from sklearn.metrics import roc_auc_score, f1_score
        yp  = ga.best_pipeline.predict(X_test)
        ypp = ga.best_pipeline.predict_proba(X_test)[:, 1]
        return jsonify({'success': True, 'best_params': best,
                        'cv_roc_auc': round(ga.best_fitness, 4),
                        'test_roc_auc': round(float(roc_auc_score(y_test, ypp)), 4),
                        'test_f1': round(float(f1_score(y_test, yp)), 4),
                        'generations': len(ga.history), 'history': ga.history})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/batch-predict')
def batch_predict():
    return render_template('batch-predict.html')

# ── Helpers ───────────────────────────────────────────────────────────

def _generate_recommendations(p):
    if p > 0.7:
        return ["🔴 IMMEDIATE ACTION REQUIRED: High churn risk detected",
                "📞 Contact customer within 24 hours for a retention call",
                "💰 Offer 20% discount for next 6 months",
                "🛡️ Promote Tech Support and Online Security for free for 3 months",
                "📅 Suggest switching to annual contract with first month free",
                "🎁 Offer loyalty rewards program enrollment with bonus points"]
    elif p > 0.4:
        return ["🟡 MEDIUM RISK: Monitor closely",
                "📧 Send personalised retention email with special offers",
                "🎁 Offer loyalty points or 10% discount on next bill",
                "📱 Recommend additional services bundle with 15% discount",
                "📞 Schedule a courtesy call to check satisfaction",
                "📊 Send customer satisfaction survey with incentive"]
    return ["🟢 LOW RISK: Customer likely to stay",
            "👍 Continue regular service and engagement",
            "🎯 Consider upsell opportunities for premium services",
            "📝 Send thank you note and request for referral",
            "🌟 Enrol in VIP customer program",
            "📱 Offer early access to new features and services"]


@app.errorhandler(404)
def not_found_error(e):
    return "Page Not Found", 404

@app.errorhandler(500)
def internal_error(e):
    return render_template('500.html'), 500


if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    print("=" * 55)
    print("🚀  ChurnGuard AI — Starting up")
    print(f"  Model    : {'✅' if model else '❌ Run model.py first'}")
    print(f"  Database : ✅ SQLite (churn_app.db)")
    print(f"  SHAP     : ✅ Ready")
    print(f"  Claude AI: {'✅ API key set' if os.environ.get('ANTHROPIC_API_KEY') else '⚠️  No key — using fallback'}")
    print(f"  GA       : ✅ Ready")
    print("=" * 55)
    app.run(debug=True, host='0.0.0.0', port=5000)

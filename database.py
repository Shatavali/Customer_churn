# database.py
"""
SQLAlchemy database layer for Customer Churn Prediction App.
Replaces prediction_log.jsonl with a proper relational database.
Tables: Prediction, CustomerSegment, AIRecommendation
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()


class Prediction(db.Model):
    """Stores every churn prediction made via the app or API."""
    __tablename__ = 'predictions'

    id               = db.Column(db.Integer, primary_key=True)
    timestamp        = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    probability      = db.Column(db.Float, nullable=False)
    prediction       = db.Column(db.Integer, nullable=False)   # 0 or 1
    risk_level       = db.Column(db.String(10), nullable=False) # Low/Medium/High
    customer_data    = db.Column(db.Text, nullable=False)       # JSON string
    shap_values      = db.Column(db.Text, nullable=True)        # JSON string
    ai_recommendation= db.Column(db.Text, nullable=True)        # Claude response

    # Key customer fields stored as columns for fast querying
    tenure           = db.Column(db.Integer, nullable=True)
    monthly_charges  = db.Column(db.Float, nullable=True)
    contract         = db.Column(db.String(30), nullable=True)
    internet_service = db.Column(db.String(30), nullable=True)

    def to_dict(self):
        return {
            'id':            self.id,
            'timestamp':     self.timestamp.isoformat(),
            'probability':   round(self.probability, 4),
            'prediction':    self.prediction,
            'risk_level':    self.risk_level,
            'customer_data': json.loads(self.customer_data),
            'tenure':        self.tenure,
            'monthly_charges': self.monthly_charges,
            'contract':      self.contract,
        }


class CustomerSegment(db.Model):
    """Stores KMeans cluster assignments and RFM analysis results."""
    __tablename__ = 'customer_segments'

    id              = db.Column(db.Integer, primary_key=True)
    run_timestamp   = db.Column(db.DateTime, default=datetime.utcnow)
    n_clusters      = db.Column(db.Integer, nullable=False)
    segment_data    = db.Column(db.Text, nullable=False)  # JSON: cluster stats
    rfm_data        = db.Column(db.Text, nullable=True)   # JSON: RFM table

    def to_dict(self):
        return {
            'id':           self.id,
            'run_timestamp': self.run_timestamp.isoformat(),
            'n_clusters':   self.n_clusters,
            'segment_data': json.loads(self.segment_data),
        }


class ModelMetric(db.Model):
    """Tracks model performance metrics over time."""
    __tablename__ = 'model_metrics'

    id          = db.Column(db.Integer, primary_key=True)
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow)
    model_name  = db.Column(db.String(50), nullable=False)
    accuracy    = db.Column(db.Float)
    precision   = db.Column(db.Float)
    recall      = db.Column(db.Float)
    f1_score    = db.Column(db.Float)
    roc_auc     = db.Column(db.Float)
    notes       = db.Column(db.Text)


def init_db(app):
    """Initialise database with the Flask app."""
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///churn_app.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
    with app.app_context():
        db.create_all()
    return db


def save_prediction(customer_data, probability, prediction,
                    shap_values=None, ai_recommendation=None):
    """Save a prediction record to the database."""
    risk_level = 'High' if probability > 0.7 else 'Medium' if probability > 0.4 else 'Low'
    record = Prediction(
        probability       = float(probability),
        prediction        = int(prediction),
        risk_level        = risk_level,
        customer_data     = json.dumps(customer_data),
        shap_values       = json.dumps(shap_values) if shap_values else None,
        ai_recommendation = ai_recommendation,
        tenure            = customer_data.get('tenure'),
        monthly_charges   = customer_data.get('MonthlyCharges'),
        contract          = customer_data.get('Contract'),
        internet_service  = customer_data.get('InternetService'),
    )
    db.session.add(record)
    db.session.commit()
    return record


def get_recent_predictions(limit=50):
    """Return the most recent predictions."""
    return Prediction.query.order_by(Prediction.timestamp.desc()).limit(limit).all()


def get_prediction_stats():
    """Calculate dashboard statistics from the database."""
    total  = Prediction.query.count()
    high   = Prediction.query.filter_by(risk_level='High').count()
    medium = Prediction.query.filter_by(risk_level='Medium').count()
    low    = Prediction.query.filter_by(risk_level='Low').count()

    from sqlalchemy import func
    avg_prob = db.session.query(func.avg(Prediction.probability)).scalar() or 0

    return {
        'total_predictions': total,
        'avg_probability':   round(float(avg_prob), 4),
        'high_risk_count':   high,
        'medium_risk_count': medium,
        'low_risk_count':    low,
    }

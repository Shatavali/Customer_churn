# shap_explainer.py
"""
SHAP Explainability Module for Customer Churn Prediction.
Generates per-prediction feature importance using TreeExplainer / KernelExplainer.
Returns structured data for waterfall chart rendering.
"""

import numpy as np
import pandas as pd
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')


def get_explainer(model, X_background=None):
    """
    Build a SHAP explainer appropriate for the model type.
    Tries TreeExplainer first (fast), falls back to KernelExplainer.
    """
    try:
        # Try to get the final classifier from the pipeline
        classifier = model.named_steps.get('classifier', None)
        if classifier is None:
            raise ValueError("No 'classifier' step found in pipeline")

        # TreeExplainer works for XGBoost, RandomForest, GradientBoosting
        explainer = shap.TreeExplainer(classifier)
        return explainer, 'tree'
    except Exception:
        pass

    # Fallback: KernelExplainer (model-agnostic, slower)
    if X_background is not None:
        background = shap.kmeans(X_background, 10)
        predict_fn = lambda x: model.predict_proba(
            pd.DataFrame(x, columns=X_background.columns)
        )[:, 1]
        explainer = shap.KernelExplainer(predict_fn, background)
        return explainer, 'kernel'

    raise RuntimeError("Could not build SHAP explainer")


def compute_shap_values(model, input_df, feature_names=None):
    """
    Compute SHAP values for a single prediction.

    Parameters
    ----------
    model     : fitted sklearn Pipeline
    input_df  : pd.DataFrame with one row (already feature-engineered)
    feature_names : list of feature names after preprocessing

    Returns
    -------
    dict with keys:
        feature_names  : list[str]
        shap_values    : list[float]
        base_value     : float  (expected model output)
        predicted_prob : float
        top_positive   : list of (feature, shap_val) pushing toward churn
        top_negative   : list of (feature, shap_val) pushing away from churn
    """
    try:
        # Get preprocessor output to pass correct data to TreeExplainer
        preprocessor = model.named_steps.get('preprocessor')
        smote_step   = model.named_steps.get('smote')
        classifier   = model.named_steps.get('classifier')

        if preprocessor is None or classifier is None:
            return _fallback_shap(model, input_df)

        # Transform input through preprocessor only
        X_transformed = preprocessor.transform(input_df)

        # Get feature names after one-hot encoding
        try:
            num_features = preprocessor.transformers_[0][2]
            cat_features = (preprocessor.transformers_[1][1]
                            .named_steps['onehot']
                            .get_feature_names_out(preprocessor.transformers_[1][2]))
            all_features = list(num_features) + list(cat_features)
        except Exception:
            all_features = [f"feature_{i}" for i in range(X_transformed.shape[1])]

        # Build explainer on classifier directly
        try:
            explainer = shap.TreeExplainer(classifier)
            sv = explainer.shap_values(X_transformed)

            # For binary classification some models return list of 2 arrays
            if isinstance(sv, list):
                sv = sv[1]  # take class-1 (churn) SHAP values

            shap_vals = sv[0].tolist() if hasattr(sv[0], 'tolist') else list(sv[0])
            base_val  = float(explainer.expected_value[1]
                              if hasattr(explainer.expected_value, '__len__')
                              else explainer.expected_value)

        except Exception:
            # KernelExplainer fallback on first 50 transformed features
            background = np.zeros((1, X_transformed.shape[1]))
            explainer  = shap.KernelExplainer(
                lambda x: classifier.predict_proba(x)[:, 1],
                background, silent=True
            )
            sv       = explainer.shap_values(X_transformed, nsamples=50)
            shap_vals = sv[0].tolist()
            base_val  = float(explainer.expected_value)

        # Pair features with their SHAP values
        pairs = list(zip(all_features, shap_vals))

        # Top contributors toward churn (positive SHAP)
        top_pos = sorted(
            [(f, v) for f, v in pairs if v > 0],
            key=lambda x: x[1], reverse=True
        )[:8]

        # Top contributors away from churn (negative SHAP)
        top_neg = sorted(
            [(f, v) for f, v in pairs if v < 0],
            key=lambda x: x[1]
        )[:5]

        return {
            'feature_names': all_features[:len(shap_vals)],
            'shap_values':   shap_vals,
            'base_value':    base_val,
            'top_positive':  [(f, round(v, 4)) for f, v in top_pos],
            'top_negative':  [(f, round(v, 4)) for f, v in top_neg],
            'success':       True,
        }

    except Exception as e:
        return {'success': False, 'error': str(e), 'top_positive': [], 'top_negative': []}


def _fallback_shap(model, input_df):
    """Fallback: use predict_proba perturbation when pipeline structure is opaque."""
    base_prob = model.predict_proba(input_df)[0, 1]
    importances = []
    for col in input_df.columns:
        perturbed = input_df.copy()
        if input_df[col].dtype in [np.float64, np.int64]:
            perturbed[col] = 0
        else:
            perturbed[col] = 'No'
        try:
            delta = base_prob - model.predict_proba(perturbed)[0, 1]
            importances.append((col, round(float(delta), 4)))
        except Exception:
            importances.append((col, 0.0))

    importances.sort(key=lambda x: abs(x[1]), reverse=True)
    top_pos = [(f, v) for f, v in importances if v > 0][:8]
    top_neg = [(f, v) for f, v in importances if v < 0][:5]
    return {
        'feature_names': [f for f, _ in importances],
        'shap_values':   [v for _, v in importances],
        'base_value':    0.5,
        'top_positive':  top_pos,
        'top_negative':  top_neg,
        'success':       True,
    }


def format_feature_name(name):
    """Make feature names human-readable for display."""
    replacements = {
        'tenure': 'Tenure (months)',
        'MonthlyCharges': 'Monthly Charges',
        'TotalCharges': 'Total Charges',
        'AvgMonthlySpend': 'Avg Monthly Spend',
        'ServiceCount': 'Number of Services',
        'SeniorCitizen': 'Senior Citizen',
        'Contract_Month-to-month': 'Month-to-Month Contract',
        'Contract_One year': 'One Year Contract',
        'Contract_Two year': 'Two Year Contract',
        'InternetService_Fiber optic': 'Fiber Optic Internet',
        'PaymentMethod_Electronic check': 'Electronic Check Payment',
    }
    return replacements.get(name, name.replace('_', ' ').replace('  ', ' ').title())

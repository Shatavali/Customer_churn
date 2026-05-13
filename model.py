# save_enhanced_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_selection import SelectFromModel
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🚀 Loading and preparing data...")
# Load and prepare data
df = pd.read_csv("Telco_Churn_Cleaned.csv")
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)
# Clean TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Encode target
le = LabelEncoder()
df['Churn'] = le.fit_transform(df['Churn'])

# Feature engineering function
def add_features(df):
    df = df.copy()
    
    # Basic features
    df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['tenure_years'] = df['tenure'] / 12
    
    # Service count feature
    service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    def count_services(row):
        count = 0
        for col in service_cols:
            if col in row.index:

                val = row[col]

                if val != 'No' and \
                   val != 'No phone service' and \
                     val != 'No internet service':
                     count += 1
        return count
    
    df['ServiceCount'] = df.apply(count_services, axis=1)
    
    # Additional engineered features
    df['Tenure_Group'] = pd.cut(df['tenure'], bins=[0, 6, 12, 24, 48, 100], 
                                 labels=['0-6', '7-12', '13-24', '25-48', '49+'])
    
    df['Charges_per_Service'] = df['MonthlyCharges'] / (df['ServiceCount'] + 1)
    
    # Interaction features
    if 'Contract' in df.columns:
      df['Tenure_Contract'] = df['tenure'] * df['Contract'].map({
        'Month-to-month': 1,
        'One year': 12,
        'Two year': 24
    })
    else:
      df['Tenure_Contract'] = df['tenure']
    
    return df

# Save feature engineering function
import inspect
with open('feature_utils_enhanced.py', 'w') as f:
    f.write(inspect.getsource(add_features))

print("📊 Engineering features...")
# Prepare features
X = df.drop('Churn', axis=1)
y = df['Churn']

# Add engineered features
X = add_features(X)
# Fill missing values
for col in X.columns:

    # Handle categorical columns
    if str(X[col].dtype) == 'category':

        X[col] = X[col].cat.add_categories(['Unknown'])
        X[col] = X[col].fillna('Unknown')

    # Handle object/text columns
    elif X[col].dtype == 'object':

        X[col] = X[col].fillna('Unknown')

    # Handle numerical columns
    else:

        X[col] = X[col].fillna(X[col].median())
    # Handle numerical columns
else:
    X[col] = X[col].fillna(X[col].median())
# Convert Tenure_Group to string (it's a categorical)
X['Tenure_Group'] = X['Tenure_Group'].astype(str)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Define columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"Numerical features: {len(numerical_cols)}")
print(f"Categorical features: {len(categorical_cols)}")

# Create preprocessing pipeline
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define multiple models with hyperparameters for tuning
print("\n🤖 Setting up multiple models...")

# Model 1: Logistic Regression (baseline)
logistic_params = {
    'classifier__C': [0.1, 1.0, 10.0],
    'classifier__penalty': ['l2'],
    'classifier__solver': ['liblinear', 'lbfgs']
}

# Model 2: Random Forest
rf_params = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [10, 20, 30, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Model 3: Gradient Boosting
gb_params = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__subsample': [0.8, 1.0]
}

# Model 4: XGBoost
xgb_params = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__subsample': [0.8, 1.0],
    'classifier__colsample_bytree': [0.8, 1.0]
}

# Model 5: Neural Network
mlp_params = {
    'classifier__hidden_layer_sizes': [(50,), (100,), (50, 25)],
    'classifier__activation': ['relu', 'tanh'],
    'classifier__alpha': [0.0001, 0.001, 0.01],
    'classifier__learning_rate': ['constant', 'adaptive']
}

# Create base models
base_models = {
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
    'MLPClassifier': MLPClassifier(random_state=42, max_iter=1000)
}

# Create parameter grids
param_grids = {
    'LogisticRegression': logistic_params,
    'RandomForest': rf_params,
    'GradientBoosting': gb_params,
    'XGBoost': xgb_params,
    'MLPClassifier': mlp_params
}

# Train and evaluate multiple models
print("\n🏋️ Training and evaluating multiple models...")
best_models = {}
cv_scores = {}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in base_models.items():
    print(f"\n📈 Training {name}...")
    
    # Create pipeline
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])
    
    # Grid search for hyperparameter tuning
    grid_search = GridSearchCV(
        pipeline, 
        param_grids[name], 
        cv=skf, 
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Store best model
    best_models[name] = grid_search.best_estimator_
    
    # Evaluate
    y_pred = grid_search.predict(X_test)
    y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    cv_scores[name] = {
        'Best_Params': grid_search.best_params_,
        'Best_Score': grid_search.best_score_,
        'Test_Accuracy': accuracy,
        'Test_Precision': precision,
        'Test_Recall': recall,
        'Test_F1': f1,
        'Test_ROC_AUC': roc_auc
    }
    
    print(f"  Best CV Score: {grid_search.best_score_:.4f}")
    print(f"  Test ROC-AUC: {roc_auc:.4f}")
    print(f"  Test F1-Score: {f1:.4f}")

# Create ensemble models
print("\n🤝 Creating ensemble models...")

# 1. Voting Classifier (Hard and Soft)
voting_estimators = [(name, model) for name, model in best_models.items()]

voting_clf_hard = VotingClassifier(
    estimators=voting_estimators,
    voting='hard'
)

voting_clf_soft = VotingClassifier(
    estimators=voting_estimators,
    voting='soft'
)

# 2. Stacking Classifier
stacking_clf = StackingClassifier(
    estimators=voting_estimators,
    final_estimator=LogisticRegression(random_state=42, class_weight='balanced'),
    cv=5
)

# Train ensemble models
ensemble_models = {
    'Voting_Hard': voting_clf_hard,
    'Voting_Soft': voting_clf_soft,
    'Stacking': stacking_clf
}

ensemble_scores = {}

for name, model in ensemble_models.items():
    print(f"\n📊 Training {name}...")
    
    # Create pipeline with preprocessing
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])
    
    # Fit the model
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    ensemble_scores[name] = {
        'Test_Accuracy': accuracy,
        'Test_Precision': precision,
        'Test_Recall': recall,
        'Test_F1': f1,
        'Test_ROC_AUC': roc_auc
    }
    
    print(f"  Test ROC-AUC: {roc_auc:.4f}")
    print(f"  Test F1-Score: {f1:.4f}")

# Combine all results
all_results = {**cv_scores, **ensemble_scores}

# Find best model
print("\n🏆 Identifying best model...")
best_model_name = None
best_roc_auc = 0

for name, scores in all_results.items():
    if scores['Test_ROC_AUC'] > best_roc_auc:
        best_roc_auc = scores['Test_ROC_AUC']
        best_model_name = name

print(f"\n✅ Best Model: {best_model_name}")
print(f"✅ Best ROC-AUC: {best_roc_auc:.4f}")

# Create final model with best performing approach
if best_model_name in best_models:
    final_model = best_models[best_model_name]
else:
    # For ensemble models, recreate the pipeline
    if best_model_name == 'Stacking':
        final_model = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', stacking_clf)
        ])
    elif best_model_name == 'Voting_Hard':
        final_model = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', voting_clf_hard)
        ])
    elif best_model_name == 'Voting_Soft':
        final_model = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', voting_clf_soft)
        ])
    
    final_model.fit(X_train, y_train)

# Save the best model
joblib.dump(final_model, 'best_churn_model_enhanced.pkl')
print("\n💾 Best model saved as 'best_churn_model_enhanced.pkl'")

# Save column information
column_info = {
    'numerical_cols': numerical_cols,
    'categorical_cols': categorical_cols,
    'feature_names': X.columns.tolist()
}
joblib.dump(column_info, 'column_info_enhanced.pkl')
print("✅ Column information saved")

# Save model comparison results
results_df = pd.DataFrame(all_results).T
results_df.to_csv('model_comparison_results.csv')
print("✅ Model comparison results saved to 'model_comparison_results.csv'")

# Create prediction function for easy use
def predict_churn(input_data, model=final_model):
    """
    Predict churn for new customers
    
    Parameters:
    input_data: DataFrame with customer features
    
    Returns:
    predictions, probabilities
    """
    # Add engineered features
    input_data = add_features(input_data)
    
    # Convert Tenure_Group to string
    if 'Tenure_Group' in input_data.columns:
        input_data['Tenure_Group'] = input_data['Tenure_Group'].astype(str)
    
    # Make predictions
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)
    
    return predictions, probabilities

# Save the prediction function
with open('prediction_utils.py', 'w') as f:
    f.write(inspect.getsource(predict_churn))

print("\n🎉 Enhanced model training complete!")
print("\n📊 Model Performance Summary:")
print(results_df[['Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1', 'Test_ROC_AUC']].round(4))

# ═══════════════════════════════════════════════════════════════════════
#  GENETIC ALGORITHM INTEGRATION
#  Run after standard training to find optimised XGBoost hyperparameters
# ═══════════════════════════════════════════════════════════════════════
print("\n\n" + "="*60)
print("🧬 GENETIC ALGORITHM HYPERPARAMETER OPTIMISATION")
print("="*60)

from genetic_algorithm import run_ga_optimization

# Use the same preprocessor built earlier
ga_preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('scaler', StandardScaler())]), numerical_cols),
        ('cat', Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_cols),
    ]
)

print("🔍 Running GA to evolve optimal XGBoost hyperparameters...")
print(f"   Population: 20  |  Generations: 15  |  CV folds: 3")
print(f"   Search space: {len(list(__import__('genetic_algorithm').PARAM_SPACE))} hyperparameters\n")

ga, best_params = run_ga_optimization(
    X_train, y_train,
    preprocessor=ga_preprocessor,
    population_size=20,
    n_generations=15,
    save_history=True,
)

# Evaluate GA-tuned model on test set
print("\n📊 Evaluating GA-optimised model on test set...")
y_pred_ga       = ga.best_pipeline.predict(X_test)
y_pred_proba_ga = ga.best_pipeline.predict_proba(X_test)[:, 1]

ga_metrics = {
    'Test_Accuracy':  accuracy_score(y_test, y_pred_ga),
    'Test_Precision': precision_score(y_test, y_pred_ga),
    'Test_Recall':    recall_score(y_test, y_pred_ga),
    'Test_F1':        f1_score(y_test, y_pred_ga),
    'Test_ROC_AUC':   roc_auc_score(y_test, y_pred_proba_ga),
    'Best_Params':    best_params,
    'Best_Score':     ga.best_fitness,
}

print(f"  Test ROC-AUC : {ga_metrics['Test_ROC_AUC']:.4f}")
print(f"  Test F1-Score: {ga_metrics['Test_F1']:.4f}")
print(f"  Best params  : {best_params}")

# Add to overall results
all_results['GA_XGBoost'] = ga_metrics

# Compare GA model against previous best
if ga_metrics['Test_ROC_AUC'] > best_roc_auc:
    best_roc_auc    = ga_metrics['Test_ROC_AUC']
    best_model_name = 'GA_XGBoost'
    joblib.dump(ga.best_pipeline, 'best_churn_model.pkl')
    print("\n🏆 GA model is NEW BEST — saved as best_churn_model.pkl")
else:
    print(f"\n✅ GA model evaluated (existing best: {best_model_name} @ {best_roc_auc:.4f})")
    joblib.dump(ga.best_pipeline, 'ga_xgboost_model.pkl')
    print("   GA pipeline saved as ga_xgboost_model.pkl")

print("\n📄 Final comparison including GA:")
results_df = pd.DataFrame(all_results).T
print(results_df[['Test_Accuracy', 'Test_Precision', 'Test_Recall',
                   'Test_F1', 'Test_ROC_AUC']].round(4))
results_df.to_csv('model_comparison_results.csv')
print("\n✅ Updated model_comparison_results.csv")

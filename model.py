# save_enhanced_model_optimized.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random state for reproducibility
RANDOM_STATE = 42

print("🚀 Loading and preparing data...")

# Load data efficiently
df = pd.read_csv("Telco_Churn_Cleaned.csv")

# Print column names to debug
print("📋 Available columns:", df.columns.tolist())

# Drop customerID if exists
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

# Clean TotalCharges more efficiently
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Encode target
if 'Churn' in df.columns:
    le = LabelEncoder()
    df['Churn'] = le.fit_transform(df['Churn'])
else:
    # Try to find Churn column with different case
    churn_col = next((col for col in df.columns if col.lower() == 'churn'), None)
    if churn_col:
        le = LabelEncoder()
        df['Churn'] = le.fit_transform(df[churn_col])
        df = df.drop(churn_col, axis=1)  # Remove original and keep encoded version
    else:
        raise ValueError("No 'Churn' column found in the dataset")

# Optimized feature engineering with flexible column names
def add_features_optimized(df):
    """Add engineered features efficiently using vectorized operations"""
    df = df.copy()
    
    # Try to identify service columns (case-insensitive)
    service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    # Find which service columns actually exist in the dataframe
    existing_service_cols = []
    for col in service_cols:
        # Try exact match first
        if col in df.columns:
            existing_service_cols.append(col)
        else:
            # Try case-insensitive match
            matches = [c for c in df.columns if c.lower() == col.lower()]
            if matches:
                existing_service_cols.append(matches[0])
    
    print(f"Found {len(existing_service_cols)} service columns")
    
    # Check if tenure exists
    if 'tenure' in df.columns:
        df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
        df['tenure_years'] = df['tenure'] / 12
    else:
        # Try to find tenure column
        tenure_col = next((col for col in df.columns if col.lower() == 'tenure'), None)
        if tenure_col:
            df['AvgMonthlySpend'] = df['TotalCharges'] / (df[tenure_col] + 1)
            df['tenure_years'] = df[tenure_col] / 12
        else:
            print("⚠️ Warning: 'tenure' column not found")
            df['AvgMonthlySpend'] = df['TotalCharges']
            df['tenure_years'] = 0
    
    # Service count - vectorized approach (only if service columns exist)
    if existing_service_cols:
        service_df = df[existing_service_cols].copy()
        # Convert to string if needed
        for col in existing_service_cols:
            if service_df[col].dtype.name == 'category':
                service_df[col] = service_df[col].astype(str)
            elif service_df[col].dtype != 'object':
                service_df[col] = service_df[col].astype(str)
        
        df['ServiceCount'] = (~service_df.isin(['No', 'No phone service', 'No internet service'])).sum(axis=1)
    else:
        print("⚠️ Warning: No service columns found")
        df['ServiceCount'] = 0
    
    # Tenure groups
    if 'tenure' in df.columns:
        df['Tenure_Group'] = pd.cut(df['tenure'], bins=[0, 6, 12, 24, 48, 100], 
                                     labels=['0-6', '7-12', '13-24', '25-48', '49+'])
    else:
        tenure_col = next((col for col in df.columns if col.lower() == 'tenure'), None)
        if tenure_col:
            df['Tenure_Group'] = pd.cut(df[tenure_col], bins=[0, 6, 12, 24, 48, 100], 
                                         labels=['0-6', '7-12', '13-24', '25-48', '49+'])
        else:
            df['Tenure_Group'] = '0-6'  # Default
    
    if 'MonthlyCharges' in df.columns:
        df['Charges_per_Service'] = df['MonthlyCharges'] / (df['ServiceCount'] + 1)
    else:
        monthly_col = next((col for col in df.columns if col.lower() == 'monthlycharges'), None)
        if monthly_col:
            df['Charges_per_Service'] = df[monthly_col] / (df['ServiceCount'] + 1)
        else:
            df['Charges_per_Service'] = 0
    
    # Interaction features
    contract_col = next((col for col in df.columns if col.lower() == 'contract'), None)
    if contract_col:
        contract_map = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
        df['Tenure_Contract'] = df['tenure'] * df[contract_col].map(contract_map).fillna(1)
    else:
        df['Tenure_Contract'] = df['tenure'] if 'tenure' in df.columns else 0
    
    return df

print("📊 Engineering features...")
X = df.drop('Churn', axis=1)
y = df['Churn']

# Add engineered features
X = add_features_optimized(X)

# Efficient missing value handling - handle categorical columns properly
print("\n🔄 Handling missing values...")
for col in X.columns:
    if X[col].dtype.name == 'category':
        # For categorical columns, add 'Unknown' category first
        if not pd.isna(X[col]).any():
            # No missing values, skip
            continue
        # Get current categories
        current_cats = list(X[col].cat.categories)
        if 'Unknown' not in current_cats:
            X[col] = X[col].cat.add_categories(['Unknown'])
        X[col] = X[col].fillna('Unknown')
    elif X[col].dtype == 'object':
        X[col] = X[col].fillna('Unknown')
    else:
        # Numerical columns
        X[col] = X[col].fillna(X[col].median())

# Convert Tenure_Group to string if it exists and is categorical
if 'Tenure_Group' in X.columns:
    if X['Tenure_Group'].dtype.name == 'category':
        X['Tenure_Group'] = X['Tenure_Group'].astype(str)
    else:
        X['Tenure_Group'] = X['Tenure_Group'].astype(str)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Define columns - identify which are categorical
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Numerical features: {len(numerical_cols)}")
print(f"Categorical features: {len(categorical_cols)}")
print(f"Sample features: {X.columns.tolist()[:10]}...")

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

# Define models with reduced hyperparameter search space for faster training
print("\n🤖 Setting up multiple models...")

# Reduce parameter grids for faster training
param_grids = {
    'LogisticRegression': {
        'classifier__C': [0.1, 1.0],
        'classifier__solver': ['lbfgs']
    },
    'RandomForest': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20],
        'classifier__min_samples_split': [2, 5]
    },
    'GradientBoosting': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.1]
    },
    'XGBoost': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.1],
        'classifier__subsample': [0.8]
    },
    'MLPClassifier': {
        'classifier__hidden_layer_sizes': [(50,), (100,)],
        'classifier__alpha': [0.001]
    }
}

# Create base models with reduced complexity
base_models = {
    'LogisticRegression': LogisticRegression(random_state=RANDOM_STATE, max_iter=500, class_weight='balanced'),
    'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
    'XGBoost': XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', use_label_encoder=False, n_jobs=-1),
    'MLPClassifier': MLPClassifier(random_state=RANDOM_STATE, max_iter=500)
}

# Train models with limited CV folds
print("\n🏋️ Training and evaluating models...")
best_models = {}
cv_scores = {}
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

for name, model in base_models.items():
    print(f"\n📈 Training {name}...")
    
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('classifier', model)
    ])
    
    # Use reduced grid search
    grid_search = GridSearchCV(
        pipeline, 
        param_grids[name], 
        cv=skf, 
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
    
    y_pred = grid_search.predict(X_test)
    y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
    
    cv_scores[name] = {
        'Best_Params': grid_search.best_params_,
        'Test_Accuracy': accuracy_score(y_test, y_pred),
        'Test_Precision': precision_score(y_test, y_pred),
        'Test_Recall': recall_score(y_test, y_pred),
        'Test_F1': f1_score(y_test, y_pred),
        'Test_ROC_AUC': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"  Test F1: {cv_scores[name]['Test_F1']:.4f}, ROC-AUC: {cv_scores[name]['Test_ROC_AUC']:.4f}")

# Create ensemble models
print("\n🤝 Creating ensemble models...")

# Preprocess the data once for ensemble models
print("Preprocessing data for ensemble models...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Apply SMOTE on processed data
smote = SMOTE(random_state=RANDOM_STATE)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

# Create base models for ensemble (without preprocessing)
ensemble_base_models = {
    'LogisticRegression': LogisticRegression(random_state=RANDOM_STATE, max_iter=500, class_weight='balanced'),
    'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
    'XGBoost': XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', use_label_encoder=False, n_jobs=-1),
    'MLPClassifier': MLPClassifier(random_state=RANDOM_STATE, max_iter=500)
}

# Train models on processed data for ensemble
trained_ensemble_models = {}
for name, model in ensemble_base_models.items():
    print(f"  Training {name} for ensemble...")
    model.fit(X_train_resampled, y_train_resampled)
    trained_ensemble_models[name] = model

# Create voting and stacking classifiers
voting_estimators = [(name, model) for name, model in trained_ensemble_models.items()]

voting_clf = VotingClassifier(estimators=voting_estimators, voting='soft')
stacking_clf = StackingClassifier(
    estimators=voting_estimators,
    final_estimator=LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced'),
    cv=3
)

ensemble_models = {
    'Voting_Soft': voting_clf,
    'Stacking': stacking_clf
}

ensemble_scores = {}

for name, model in ensemble_models.items():
    print(f"\n📊 Training {name}...")
    
    model.fit(X_train_resampled, y_train_resampled)
    
    y_pred = model.predict(X_test_processed)
    y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    
    ensemble_scores[name] = {
        'Test_Accuracy': accuracy_score(y_test, y_pred),
        'Test_Precision': precision_score(y_test, y_pred),
        'Test_Recall': recall_score(y_test, y_pred),
        'Test_F1': f1_score(y_test, y_pred),
        'Test_ROC_AUC': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"  Test F1: {ensemble_scores[name]['Test_F1']:.4f}, ROC-AUC: {ensemble_scores[name]['Test_ROC_AUC']:.4f}")

# Combine results
all_results = {**cv_scores, **ensemble_scores}

# Find best model
best_model_name = max(all_results, key=lambda x: all_results[x]['Test_ROC_AUC'])
best_roc_auc = all_results[best_model_name]['Test_ROC_AUC']

print(f"\n✅ Best Model: {best_model_name}")
print(f"✅ Best ROC-AUC: {best_roc_auc:.4f}")

# Create and save final model
final_model = None

if best_model_name in cv_scores:
    # It's a base model - create a full pipeline
    print(f"\n🔧 Creating final pipeline for {best_model_name}...")
    final_model = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('classifier', base_models[best_model_name])
    ])
    final_model.fit(X_train, y_train)
    
elif best_model_name in ensemble_scores:
    # It's an ensemble model - create wrapper with preprocessing
    print(f"\n🔧 Creating final ensemble wrapper for {best_model_name}...")
    
    class EnsembleWrapper:
        def __init__(self, ensemble_model, preprocessor):
            self.ensemble_model = ensemble_model
            self.preprocessor = preprocessor
            self.fitted = False
        
        def fit(self, X, y):
            X_processed = self.preprocessor.fit_transform(X)
            # Apply SMOTE
            smote = SMOTE(random_state=RANDOM_STATE)
            X_resampled, y_resampled = smote.fit_resample(X_processed, y)
            self.ensemble_model.fit(X_resampled, y_resampled)
            self.fitted = True
            return self
        
        def predict(self, X):
            if not self.fitted:
                raise ValueError("Model must be fitted before prediction")
            X_processed = self.preprocessor.transform(X)
            return self.ensemble_model.predict(X_processed)
        
        def predict_proba(self, X):
            if not self.fitted:
                raise ValueError("Model must be fitted before prediction")
            X_processed = self.preprocessor.transform(X)
            return self.ensemble_model.predict_proba(X_processed)
    
    final_model = EnsembleWrapper(ensemble_models[best_model_name], preprocessor)
    final_model.fit(X_train, y_train)

# Save the final model
if final_model is not None:
    joblib.dump(final_model, 'best_churn_model.pkl')
    print("\n💾 Best model saved as 'best_churn_model.pkl'")
else:
    print("\n⚠️ Warning: No final model created!")

# Save column info
column_info = {
    'numerical_cols': numerical_cols,
    'categorical_cols': categorical_cols,
    'feature_names': X.columns.tolist(),
    'preprocessor': preprocessor
}
joblib.dump(column_info, 'column_info.pkl')
print("✅ Column information saved")

# Save results
results_df = pd.DataFrame(all_results).T
results_df.to_csv('model_comparison_results.csv')
print("✅ Model comparison results saved")

# Simple prediction function
def predict_churn(input_data):
    """Predict churn for new customers"""
    if final_model is None:
        raise ValueError("No model available for prediction")
    
    input_data = add_features_optimized(input_data)
    if 'Tenure_Group' in input_data.columns:
        if input_data['Tenure_Group'].dtype.name == 'category':
            input_data['Tenure_Group'] = input_data['Tenure_Group'].astype(str)
        else:
            input_data['Tenure_Group'] = input_data['Tenure_Group'].astype(str)
    
    predictions = final_model.predict(input_data)
    probabilities = final_model.predict_proba(input_data)
    return predictions, probabilities

print("\n🎉 Model training complete!")
print("\n📊 Performance Summary:")
print(results_df[['Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1', 'Test_ROC_AUC']].round(4))

print(f"\n🏆 Best performing model: {best_model_name} with ROC-AUC: {best_roc_auc:.4f}")
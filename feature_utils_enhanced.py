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

def add_features(df):
    df = df.copy()
    df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['tenure_years'] = df['tenure'] / 12
    
    service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['ServiceCount'] = df[service_cols].apply(
        lambda row: sum([1 for val in row if val != 'No' and val != 'No phone service' and val != 'No internet service']), 
        axis=1
    )
    return df

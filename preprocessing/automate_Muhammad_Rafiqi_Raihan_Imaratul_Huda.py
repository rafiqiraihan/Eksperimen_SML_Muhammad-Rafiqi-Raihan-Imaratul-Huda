from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# Pre Cleaning
def clean_telco_data(data):
    data = data.copy()

    # Drop kolom ID
    data.drop(columns=['customerID'], inplace=True)

    # Konversi TotalCharges ke numerik
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

    return data

# Preprocesing pipeline
def preprocesing_data(data, target_column, save_path, file_path):
    # Menentukan fitur numerik dan kategorikal
    numeric_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()

    # Menghapus target dari daftar fitur
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)

    # Simpan nama kolom target
    column_names = data.drop(columns=[target_column]).columns
    pd.DataFrame(columns=column_names).to_csv(file_path, index=False)
    print(f"Nama kolom berhasil disimpan ke: {file_path}")

    # Pipeline untuk numerik
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
    ])

    # Pipeline untuk kategorikal
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # ColumnTransformer gabungan
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # Split data
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit transform pada train, transform pada test
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Simpan pipeline
    dump(preprocessor, save_path)
    print(f"Preprocessing pipeline berhasil disimpan ke: {save_path}")

    # Simpan hasil preprocessing sebagai CSV
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
    
    df_out = pd.DataFrame(X_train_processed)
    df_out[target_column] = y_train.values
    df_out.to_csv(file_path, index=False)
    print(f"Dataset hasil preprocessing berhasil disimpan ke: {file_path}")

    return X_train_processed, X_test_processed, y_train, y_test

if __name__ == "__main__":
    data = pd.read_csv('Telco-Customer-Churn_raw.csv')
    data = clean_telco_data(data)
    data['is_churn'] = data['Churn'].map({'No':0, 'Yes': 1})
    data.drop(columns=['Churn'], inplace=True)

    preprocesing_data(
        data=data,
        target_column='is_churn',
        save_path='preprocessing/preprocessor.joblib',
        file_path='preprocessing/telco_preprocessing.csv'
    )

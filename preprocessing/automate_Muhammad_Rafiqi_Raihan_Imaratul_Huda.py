from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pandas as pd
import os

# Pre Cleaning
def clean_telco_data(data):
    data = data.copy()

    # Drop kolom ID
    data.drop(columns=['customerID'], inplace=True)

    # Konversi TotalCharges ke numerik
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].mean())

    return data

# Preprocessing pipeline
def preprocesing_data(data, target_column, save_path, file_path):
    # Menentukan fitur numerik dan kategorikal
    numeric_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()

    # Hapus kolom target dari daftar fitur
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)

    # Pipeline numerik
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
    ])

    # Pipeline kategorikal
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Gabungkan pipeline
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # Split X dan y
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Fit dan transform seluruh dataset
    X_processed = preprocessor.fit_transform(X)

    # Ambil nama kolom hasil encoding
    encoded_cat_columns = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features)
    all_columns = numeric_features + list(encoded_cat_columns)

    # Simpan pipeline
    dump(preprocessor, save_path)
    print(f"Preprocessing pipeline berhasil disimpan ke: {save_path}")

    # Simpan sebagai CSV
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    data_processed = pd.DataFrame(X_processed, columns=all_columns)
    data_processed[target_column] = y.values
    data_processed.to_csv(file_path, index=False)
    print(f"Dataset hasil preprocessing berhasil disimpan ke: {file_path}")

    return data_processed

if __name__ == "__main__":
    data = pd.read_csv('Telco-Customer-Churn_raw.csv')
    data = clean_telco_data(data)
    data['is_churn'] = data['Churn'].map({'No': 0, 'Yes': 1})
    data.drop(columns=['Churn'], inplace=True)

    preprocesing_data(
        data=data,
        target_column='is_churn',
        save_path='preprocessing/preprocessor.joblib',
        file_path='preprocessing/telco_preprocessing.csv'
    )

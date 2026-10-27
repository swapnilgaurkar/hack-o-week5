import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def get_feature_columns(df):
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()
    return numeric_columns, categorical_columns


def prepare_dataframe(df):
    numeric_columns, categorical_columns = get_feature_columns(df)
    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns
    }


def standardized_numeric_data(df):
    numeric_columns, _ = get_feature_columns(df)

    if not numeric_columns:
        raise ValueError("No numerical features are available for dimensionality reduction.")

    numeric = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

    # Drop columns that contain no usable numeric value at all.
    usable_columns = numeric.columns[numeric.notna().any()].tolist()
    numeric = numeric[usable_columns]

    if numeric.shape[1] == 0:
        raise ValueError("No usable numerical features were found.")

    # Median imputation handles missing numerical values.
    imputer = SimpleImputer(strategy="median")
    filled = imputer.fit_transform(numeric)

    if not np.isfinite(filled).all():
        raise ValueError("The dataset contains values that could not be processed.")

    # Standardization puts features on a comparable scale.
    scaler = StandardScaler()
    scaled = scaler.fit_transform(filled)

    return scaled, usable_columns


def dataset_summary(df):
    missing_total = int(df.isna().sum().sum())
    missing_by_column = {
        str(k): int(v)
        for k, v in df.isna().sum().items()
        if int(v) > 0
    }

    numeric_columns, categorical_columns = get_feature_columns(df)

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_features": len(numeric_columns),
        "categorical_features": len(categorical_columns),
        "missing_values": missing_total,
        "missing_by_column": missing_by_column
    }


def get_class_values(df, class_column):
    if not class_column or class_column not in df.columns:
        return None

    values = df[class_column].fillna("Missing").astype(str)
    return values.tolist()

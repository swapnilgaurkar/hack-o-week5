import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History"
]

CATEGORICAL_FEATURES = [
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "Property_Area"
]

# Global fitted transformer used for new prediction data.
preprocessor = None

def load_and_preprocess_data(path):
    global preprocessor

    df = pd.read_csv(path)
    df["Dependents"] = df["Dependents"].replace("3+", "3")

    X = df.drop(columns=["Loan_ID", "Loan_Status"])
    y = df["Loan_Status"].map({"N": 0, "Y": 1})

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    feature_names = list(preprocessor.get_feature_names_out())
    return X_train, X_test, y_train, y_test, feature_names

def preprocess_new_data(df):
    if preprocessor is None:
        raise RuntimeError("Preprocessor has not been fitted.")
    df = df.copy()
    df["Dependents"] = df["Dependents"].replace("3+", "3")
    return preprocessor.transform(df)
